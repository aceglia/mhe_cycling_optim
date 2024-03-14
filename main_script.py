"""
This script is the main script for the project. It is used to run the mhe solver and visualize the estimated data.
"""
import os.path
import shutil
import multiprocessing as mp
from mhe.ocp import *
from mhe.utils import *
from pathlib import Path
from biosiglive import MskFunctions, InverseKinematicsMethods


class MuscleForceEstimator:
    """
    This class is used to define the muscle force estimator.
    """

    def __init__(self, *args):
        """
        Initialize the muscle force estimator.

        Parameters
        ----------
        args : dict
            Dictionary of configuration to initialize the estimator.
        """
        conf = check_and_adjust_dim(*args)
        self.model_path = conf["model_path"]
        biorbd_model = BiorbdModel(self.model_path)
        self.use_torque = False
        self.save_results = True
        self.track_emg = False
        self.data_to_show = [""]
        self.kin_data_to_track = None
        self.test_offline = False
        self.offline_file = None
        self.plot_delay = []

        # Variables of the problem
        self.exp_freq = 35
        self.ns_mhe = 0
        self.mhe_time = 0.1
        self.markers_rate = 100
        self.emg_rate = 2000
        self.get_names = False
        self.get_kalman = True
        self.offline_data = None
        self.muscle_track_idx = []
        self.solver_options = {}
        self.message = None
        self.vicon_client = None
        self.var, self.server_ip, self.server_port, self.data_to_show = {}, None, None, []

        # multiprocess stuffs
        manager = mp.Manager()
        self.data_count = mp.Value("i", 0)
        self.plot_queue = manager.Queue()
        self.data_queue = manager.Queue()
        self.data_event = mp.Event()
        self.process = mp.Process
        self.plot_event = mp.Event()

        self.p_q, self.win_q, self.app_q, self.box_q = [], [], [], []
        self.p_force, self.win_force, self.app_force, self.box_force = [], [], [], []
        self.plot_force_ratio, self.plot_q_ratio = 0, 0
        self.print_lvl = 1
        self.plot_q_freq, self.plot_force_freq = self.exp_freq, 10
        self.force_to_plot, self.q_to_plot = [], []
        self.count_p_f, self.count_p_q = [], []
        self.mvc_list = None
        self.interpol_factor = 1
        self.weights = {}
        self.result_dir = None
        self.ns_full = None
        self.init_n = 0
        self.final_n = None
        self.result_file_name = None
        self.markers_target, self.muscles_target, self.x_ref, self.kin_target, self.f_ext_target = None, None, None, None, None
        self.n_loop = 0
        self.mhe, self.solver, self.get_force, self.force_est = None, None, None, None
        self.model = None
        self.b = None
        self.frame_to_save = 0
        self.save_all_frame = True
        self.with_f_ext = False
        self.f_ext_as_constraints = False
        self.emg_names = []

        # Use the configuration dictionary to initialize the muscle force estimator parameters
        for key in conf.keys():
            self.__dict__[key] = conf[key]

        if self.f_ext_as_constraints and self.with_f_ext is False:
            raise RuntimeError(
                "we must have with_f_ext True to use constraints"
            )

        self.T_mhe = self.mhe_time
        self.ns_mhe = int(self.T_mhe * self.markers_rate * self.interpol_factor)
        self.slide_size = int(((self.markers_rate * self.interpol_factor) / self.exp_freq))
        self.nbQ, self.nbMT = biorbd_model.nb_q, biorbd_model.nb_muscles
        self.nbGT = biorbd_model.nb_tau if self.use_torque else 0
        self.current_time = strftime("%Y%m%d-%H%M")
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")

        self.markers_ratio = 1
        self.EMG_ratio = 1
        self.rt_ratio = self.markers_ratio
        self.muscle_names = []
        for i in range(biorbd_model.nb_muscles):
            self.muscle_names.append(biorbd_model.muscle_names[i])
        self.dof_names = []
        for i in range(biorbd_model.nb_q):
            self.dof_names.append(biorbd_model.name_dof[i])

    def prepare_problem_init(self):
        """
        Prepare the mhe problem.
        """

        biorbd_model = BiorbdModel(self.model_path)
        markers_target, markers_names, forces_object, f_ext_target, emg = load_data(self.offline_file,
                                                                                     win_size=self.ns_mhe,
                                                               filter_depth=False
                                                           )
        forces_object = biorbd_model.model.externalForceSet()
        f_ext_target = f_ext_target[:, 0, :]

        msk_function = MskFunctions(model=self.model_path, data_buffer_size=self.ns_mhe//self.interpol_factor + 1)
        x_ref = msk_function.compute_inverse_kinematics(markers_target[..., :self.ns_mhe//self.interpol_factor + 1],
                                                        method=InverseKinematicsMethods.BiorbdKalman)[0]

        self.offline_data = [x_ref, markers_target, emg, forces_object, f_ext_target]

        window_len = self.ns_mhe
        window_duration = self.T_mhe
        self.muscle_track_idx = get_tracking_idx(msk_function.model, self.emg_names)
        if emg is not None:
            muscles_target = map_activation(
                emg_proc=emg, muscle_track_idx=self.muscle_track_idx ,
                model=msk_function.model,
                emg_names=self.emg_names)
        else:
            muscles_target = np.zeros((biorbd_model.nb_muscles, window_len // self.interpol_factor ))
        self.x_ref, self.markers_target, self.muscles_target, self.f_ext_target = interpolate_data(
            self.interpol_factor, x_ref, muscles_target, markers_target, f_ext_target,
        )
        self.markers_target = self.markers_target[:, :, :window_len]

        # self.f_ext_target = self.f_ext_target.T[:, :window_len, 0]
        self.kin_target = (
            self.markers_target[:, :, : window_len + 1]
            if self.kin_data_to_track == "markers"
            else self.x_ref[: self.nbQ, : window_len + 1]
        )

        for i in range(biorbd_model.nb_muscles):
            self.muscle_names.append(biorbd_model.muscle_names[i])
        if self.x_ref.shape[0] != biorbd_model.nb_q * 2:
            previous_sol = np.concatenate(
                (self.x_ref[:, : window_len + 1], np.zeros((self.x_ref.shape[0], window_len + 1)))
            )
        else:
            previous_sol = self.x_ref[:, : window_len + 1]
        muscle_init = np.ones((biorbd_model.nb_muscles, self.ns_mhe)) * 0.1
        count = 0
        for i in self.muscle_track_idx:
            muscle_init[i, :] = self.muscles_target[count, : self.ns_mhe]
            count += 1
        f_ext_init = np.zeros((6, self.ns_mhe))
        f_ext_init[:, :] = self.f_ext_target[:, : self.ns_mhe]
        if self.with_f_ext:
            u0 = np.concatenate((muscle_init, np.zeros((biorbd_model.nb_q, self.ns_mhe)), f_ext_init))
        else:
            u0 = np.concatenate((muscle_init, np.zeros((biorbd_model.nb_q, self.ns_mhe))))
        objectives = define_objective(
            weights=self.weights,
            use_torque=self.use_torque,
            with_f_ext=self.with_f_ext,
            f_ext_as_constraints=self.f_ext_as_constraints,
            track_emg=self.track_emg,
            muscles_target=self.muscles_target[:, : self.ns_mhe],
            f_ext_target=self.f_ext_target,
            kin_target=self.kin_target,
            biorbd_model=biorbd_model,
            previous_sol=previous_sol,
            kin_data_to_track=self.kin_data_to_track,
            muscle_track_idx=self.muscle_track_idx,
        )

        constraints = define_constraint(
            f_ext_target=self.f_ext_target,
            with_f_ext=self.with_f_ext,
            f_ext_as_constraints=self.f_ext_as_constraints,
        )

        self.mhe, self.solver = prepare_problem(
            self.model_path,
            objectives,
            constraints,
            window_len=window_len,
            window_duration=window_duration,
            x0=self.x_ref,
            u0=u0,
            f_ext_0=self.f_ext_target,
            f_ext_object=forces_object,
            use_torque=self.use_torque,
            with_f_ext=self.with_f_ext,
            f_ext_as_constraints=self.f_ext_as_constraints,
            nb_threads=8,
            solver_options=self.solver_options,
            use_acados=True,
        )
        self.get_force = force_func(biorbd_model)
        self.force_est = np.ndarray((biorbd_model.nb_muscles, 1))

    def run(
        self,
        var: dict,
        server_ip: str,
        server_port: int,
        data_to_show: list = None,
        test_offline: bool = False,
        offline_file: str = None,
    ):
        """
        Run the whole multiprocess program.

        Parameters
        ----------
        var : dict
            Dictionary containing the parameters of the problem.
        server_ip : str
            IP of the vicon server.
        server_port : int
            Port of the vicon server.
        data_to_show : list, optional
            List of data to show. The default is None.
        test_offline : bool, optional
            If True, the program will run in offline mode. The default is False.
        offline_file : str, optional
            Path to the offline file. The default is None.
        """
        self.var = var
        self.server_ip = server_ip
        self.server_port = server_port
        self.data_to_show = data_to_show
        self.test_offline = test_offline
        self.offline_file = offline_file
        proc_plot = None
        if self.test_offline and not self.offline_file:
            raise RuntimeError("Please provide a data file to run offline program")

        if self.data_to_show:
            proc_plot = self.process(name="plot", target=MuscleForceEstimator.run_plot, args=(self,))
            proc_plot.start()

        proc_mhe = self.process(name="mhe", target=MuscleForceEstimator.run_mhe, args=(self, var, data_to_show))
        proc_mhe.start()
        if self.data_to_show:
            proc_plot.join()
        proc_mhe.join()

    def run_plot(self):
        """
        Run the plot function.
        """
        data = None
        self.all_plot = LivePlot()
        for data_to_show in self.data_to_show:
            if data_to_show == "force":
                self.all_plot.add_new_plot(
                    plot_name="Muscle force",
                    plot_type="progress_bar",
                    nb_subplot=self.nbMT,
                    channel_names=self.muscle_names,
                    unit="N",
                )
                self.rplt_force, self.layout_force, self.app_force = self.all_plot.init_plot_window(
                    self.all_plot.plot[0]
                )
            if data_to_show == "q":
                self.all_plot.msk_model = self.model_path
                self.all_plot.add_new_plot(plot_type="skeleton")
                self.all_plot.set_skeleton_plot_options(show_floor=False)
                n_plot = 0 if not "force" in self.data_to_show else 1
                self.all_plot.init_plot_window(self.all_plot.plot[n_plot])

        self.q_to_plot = np.zeros((self.nbQ, self.plot_q_ratio))
        self.plot_q_ratio = int(self.exp_freq / self.plot_q_freq)
        self.plot_force_ratio = int(self.exp_freq / self.plot_force_freq)
        self.force_to_plot = np.zeros((self.nbMT, self.plot_force_ratio))
        self.count_p_f, self.count_p_q = self.plot_force_ratio, self.plot_q_ratio
        self.plot_event.set()
        while True:
            try:
                data = self.plot_queue.get_nowait()
                is_working = True
            except:
                is_working = False
            if is_working:
                plot_delay = update_plot(self, data["force_est"], data["q_est"], init_time=data["init_time_frame"])
                dic = {"plot_delay": plot_delay}
                save_results(dic, self.current_time, result_dir=self.result_dir, file_name_prefix="plot_delay_")

    def run_mhe(self, var: dict, data_to_show: list):
        """
        Run the mhe solver.

        Parameters
        ----------
        var : dict
            Dictionary containing the parameters of the problem.
        data_to_show : list
            List of data to show.
        """
        if os.path.isdir("c_generated_code"):
            shutil.rmtree("c_generated_code")
        self.prepare_problem_init()
        if data_to_show:
            self.plot_event.wait()
        for key in var.keys():
            if key in self.__dict__:
                self.__setattr__(key, var[key])
            else:
                raise RuntimeError(f"{key} is not a variable of the class")

        self.model = BiorbdModel(self.model_path)
        initial_time = time()
        sol = self.mhe.solve(
            lambda mhe, i, sol: update_mhe(
                mhe, i, sol, self, initial_time=initial_time, offline_data=self.offline_data
            ),
            export_options={"frame_to_export": self.frame_to_save},
            solver=self.solver,
        )


if __name__ == "__main__":
    # idx_trial = 3
    # data_dir = f"/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/data_gen/saves/"
    # result_dir = "results/results_w9"
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/process_data"

    participants = ["P10"]
    init_trials = [["gear_10"]] * len(participants)
    processed_source = ["depth"]
    final_files = []
    for p, part in enumerate(participants):
        model_dir = f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models"
        result_dir = f"results/{part}"
        all_files = os.listdir(f"{data_dir}/{part}")
        all_files = [file for file in all_files if "gear" in file and "result_biomech" not in file and "3_crops" in file]
        for file in all_files:
            for trial in init_trials[participants.index(part)]:
                if trial in file:
                    final_files.append(f"{data_dir}/{part}/{file}")

        # configs = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
        # exp_freq = [43, 38, 37, 34, 29, 27, 25, 24, 22]
        configs = [0.08]
        exp_freq = [30]
        for c, config in enumerate(configs):
            for t, trial in enumerate(final_files):
                offline_path = trial
                if not os.path.isdir(result_dir):
                    os.makedirs(result_dir)
                solver_options = {
                    "sim_method_jac_reuse": 1,
                    "levenberg_marquardt": 50.0,
                    "nlp_solver_step_length": 0.9,
                    "qp_solver_iter_max": 1000,
                }

                model = f"{model_dir}/{init_trials[p][t]}_processed_3_model_scaled_depth.bioMod"
                configuration_dic = {
                    "model_path": model,
                    "mhe_time": config,
                    "interpol_factor": 2,
                    "use_torque": True,
                    "save_results": True,
                    "track_emg": True,
                    "with_f_ext": True,
                    "f_ext_as_constraints": False,
                    "kin_data_to_track": "markers",
                    "exp_freq": exp_freq[c],
                    "result_dir": result_dir,
                    "result_file_name": f"result_mhe_{trial}",
                    "solver_options": solver_options,
                    "weights": configure_weights(),
                    "frame_to_save": 0,
                    "save_all_frame": False,
                    "emg_names": ["PECM",
                                  "bic",
                                  "tri",
                                  "LAT",
                                  'TRP1',
                                  "DELT1",
                                  'DELT2',
                                  'DELT3']
                }
                variables_dic = {"print_lvl": 0}  # print level 0 = no print, 1 = print information
                data_to_show = None  # ["q", "force"]
                server_ip = "192.168.1.211"
                server_port = 50000
                MHE = MuscleForceEstimator(configuration_dic)
                MHE.run(variables_dic, server_ip, server_port, data_to_show, test_offline=True, offline_file=offline_path)
