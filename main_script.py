"""
This script is the main script for the project. It is used to run the mhe solver and visualize the estimated data.
"""
import os.path

import biorbd
import shutil
import multiprocessing as mp

import matplotlib.pyplot as plt

from mhe.ocp import *
from mhe.utils import *
from pathlib import Path
from biosiglive import MskFunctions, InverseKinematicsMethods
import biorbd as biorbd_eigen


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
        self.biorbd_model = None
        self.use_optim_params = False
        self.parameters_file_path = None
        self.part = None
        self.use_acados = True
        self.source = None
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
        self.markers_rate = 120
        self.emg_rate = 2160
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
        self.n_before_interpolate = int(self.T_mhe * self.markers_rate)
        self.ns_mhe = int(self.T_mhe * self.markers_rate * self.interpol_factor)
        # self.slide_size = int(((self.markers_rate * self.interpol_factor) / self.exp_freq))
        self.slide_size = 1
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

    def _update_params(self, biorbd_model, parameters_file_path, with_casadi=False, ratio=True):
        with open(biorbd_model, "r") as file:
            str_model = file.read()
        param_model = biorbd_eigen.Model(biorbd_model)
        param_model = apply_params(param_model, parameters_file_path, with_casadi=with_casadi, ratio=ratio)
        start_idx = 0
        str_model_tmp = ""
        count = 0
        end_idx = 0
        while True:
            idx_tmp = str_model.find("optimalLength", start_idx) + len(r"optimalLength\t") - 1
            if idx_tmp == -1 + len(r"optimalLength\t") - 1:
                str_model_tmp += str_model[end_idx:]
                break
            str_model_tmp += str_model[end_idx:idx_tmp]
            end_idx = str_model.find('\n', idx_tmp)
            str_model_tmp += str(param_model.muscle(count).characteristics().optimalLength())
            start_idx = end_idx
            count += 1

        str_model = str_model_tmp

        start_idx = 0
        str_model_tmp = ""
        count = 0
        end_idx = 0
        while True:
            idx_tmp = str_model.find("maximalForce", start_idx) + len(r"maximalForce\t") - 1
            if idx_tmp == -1 + len(r"maximalForce\t") - 1:
                str_model_tmp += str_model[end_idx:]
                break
            str_model_tmp += str_model[end_idx:idx_tmp]
            end_idx = str_model.find('\n', idx_tmp)
            str_model_tmp += str(param_model.muscle(count).characteristics().forceIsoMax())
            start_idx = end_idx
            count += 1
        str_model = str_model_tmp
        start_idx = 0
        str_model_tmp = ""
        count = 0
        end_idx = 0
        while True:
            idx_tmp = str_model.find("tendonSlackLength", start_idx) + len(r"tendonSlackLength\t") - 1
            if idx_tmp == -1 + len(r"tendonSlackLength\t") - 1:
                str_model_tmp += str_model[end_idx:]
                break
            str_model_tmp += str_model[end_idx:idx_tmp]
            end_idx = str_model.find('\n', idx_tmp)
            str_model_tmp += str(param_model.muscle(count).characteristics().tendonSlackLength())
            start_idx = end_idx
            count += 1
        param_model = None
        with open(biorbd_model[:-7] + "_params.bioMod", "w") as file:
            file.write(str_model_tmp)
        return biorbd_model[:-7] + "_params.bioMod"

    def prepare_problem_init(self):
        """
        Prepare the mhe problem.
        """
        if self.use_optim_params:
           self.model_path = self._update_params(self.model_path, self.parameters_file_path, with_casadi=False, ratio=True)
        self.biorbd_model = BiorbdModel(self.model_path)
        # Old data :
        # x_ref, markers_target, emg = get_data(offline=True, offline_file_path=self.offline_file)
        # _, _, forces_object, f_ext_target, _ = load_data(self.offline_file,
        #                                                                              win_size=self.ns_mhe,
        #                                                                             source=self.source,
        #                                                        filter_depth=False
        #                                                    )
        # f_ext_target = np.zeros((6, 1, emg.shape[1]))
        # forces_object = None

        # New data :
        self.muscle_track_idx = get_tracking_idx(self.biorbd_model.model, self.emg_names)
        self.offline_data, self.markers_target, markers_names, self.f_ext_target, self.muscles_target, self.x_ref = load_data(self.offline_file,
                                                                                                                              part=self.part,
                                                                                      source=self.source,
                                                                                      filter_depth=True,
                                                                                      model=biorbd_eigen.Model(self.model_path),
                                                                                      muscle_track_idx=self.muscle_track_idx,
                                                                                      emg_names=self.emg_names,
                                                                                      interp_factor=self.interpol_factor,
                                                                                      n_init=10,
                                                                                      n_final=None
                                                                                      )

        # if self.use_optim_params:
        #     self.biorbd_model.model = apply_params(self.biorbd_model, self.parameters_file_path, with_casadi=True, ratio=True)
        forces_object = self.biorbd_model.model.externalForceSet()
        self.kin_target = self.markers_target if self.kin_data_to_track == "markers" else self.x_ref[: self.nbQ, :]

        muscle_init = np.ones((self.biorbd_model.nb_muscles, self.ns_mhe)) * 0.1
        if self.track_emg:
            count = 0
            for i in self.muscle_track_idx:
                muscle_init[i, :] = self.muscles_target[count, : self.ns_mhe]
                count += 1

        objectives = define_objective(
            weights=self.weights,
            use_torque=self.use_torque,
            with_f_ext=self.with_f_ext,
            f_ext_as_constraints=self.f_ext_as_constraints,
            track_emg=self.track_emg,
            muscles_target=self.muscles_target[:, : self.ns_mhe],
            f_ext_target=self.f_ext_target[:, : self.ns_mhe],
            kin_target=self.kin_target[..., : self.ns_mhe + 1],
            biorbd_model=self.biorbd_model,
            previous_sol=self.x_ref[:, : self.ns_mhe + 1].copy(),
            kin_data_to_track=self.kin_data_to_track,
            muscle_track_idx=self.muscle_track_idx,
        )

        # constraints = define_constraint(
        #     f_ext_target=self.f_ext_target,
        #     with_f_ext=self.with_f_ext,
        #     f_ext_as_constraints=self.f_ext_as_constraints,
        # )
        self.mhe, self.solver = prepare_problem(
            self.model_path,
            objectives,
            # constraints,
            window_len=self.ns_mhe,
            window_duration=self.T_mhe,
            x0=self.x_ref[:, : self.ns_mhe + 1].copy(),
            f_ext_0=self.f_ext_target[:, : self.ns_mhe],
            f_ext_object=forces_object,
            use_torque=self.use_torque,
            with_f_ext=self.with_f_ext,
            f_ext_as_constraints=self.f_ext_as_constraints,
            nb_threads=8,
            solver_options=self.solver_options,
            use_acados=self.use_acados,
        )
        print(self.solver.__getattribute__("_sim_method_jac_reuse"))
        self.mhe.frame_to_export = slice(self.frame_to_save, self.frame_to_save + self.slide_size)
        self.get_force = force_func(self.biorbd_model)
        self.force_est = np.ndarray((self.biorbd_model.nb_muscles, 1))

    def run(
            self,
            var: dict,
            server_ip: str = None,
            server_port: int = None,
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
        else:
            return self.run_mhe(var, data_to_show)

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
        initial_time = time()
        sol = self.mhe.solve(
            lambda mhe, i, sol: update_mhe(
                mhe, i, sol, self, initial_time=initial_time
            ),
            export_options={"frame_to_export": self.frame_to_save},
            solver=self.solver,
        )
        return

        # x_ref = self.x_ref
        # states_tmp = sol[0]
        # controls_tmp = sol[1]
        # from plot_results import plot_results
        # plot_results(x_ref, states_tmp, controls_tmp, self.muscles_target, self.f_ext_target, self.muscle_track_idx,
        #              self.nbQ)

def _remove_root_dofs(model_path):

    with open(model_path, "r") as file:
        data = file.read()
    data = data.replace(
        "rotations xyz // thorax",
        f"//rotations xyz // thorax",
    )
    data = data.replace(
        "translations xyz // thorax",
        f"// translations xyz // thorax",
    )
    new_path = model_path.replace(".bioMod", "_static_root.bioMod")
    with open(new_path, "w") as file:
        file.write(data)
    return new_path


if __name__ == "__main__":
    # idx_trial = 3
    # data_dir = f"/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/data_gen/saves/"
    # result_dir = "results/results_w9"
    # TODO: handle nan for vicon markers
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/process_data"
    if os.name == "posix":
        prefix = "/mnt/shared"
    else:
        prefix = "Q:/"
    participants = ["P10"]
    init_trials = [["gear_20"]] * len(participants)
    processed_source = ["dlc_1"]
    processed_data_path = prefix + "/Projet_hand_bike_markerless/RGBD"
    configs = [0.08]
    exp_freq = [30]
    dlc_model = "normal_500_down_b1"
    use_optim_params = [True, False]
    #for c, config in enumerate(configs):
    #c = True
    for p, part in enumerate(participants):
        model_dir = prefix + f"/Projet_hand_bike_markerless/RGBD/{part}/models"
        parameters_file_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param.bio"
        result_dir = f"results/{part}"
        for t, trial in enumerate(init_trials[p]):
            for c in use_optim_params:
                # dir = os.listdir(processed_data_path + f"/{part}")
                # dir = [d for d in dir if trial in d][0]
                data = data_dir + f"/{part}/" + f"result_biomech_{trial}_{dlc_model}_no_root.bio"
                # offline_path = data_dir + f"{trial[p]}"
                if not os.path.isdir(result_dir):
                    os.makedirs(result_dir)
                solver_options = {
                    "sim_method_jac_reuse": 1,
                    "levenberg_marquardt": 90.0,
                    "nlp_solver_step_length": 0.9,
                    "qp_solver_iter_max": 10000,
                }
                # solver_options = {
                #     "print_info_string": "yes",
                # }
                if os.path.isfile(result_dir + os.sep + f"result_mhe_{trial}_{processed_source[0]}_optim_param_{c}.bio"):
                    os.remove(result_dir + os.sep + f"result_mhe_{trial}_{processed_source[0]}_optim_param_{c}.bio")
                model = f"{model_dir}/{trial}_model_scaled_dlc_ribs_new_seth.bioMod"
                #model = _remove_root_dofs(model)
                configuration_dic = {
                    "model_path": model,
                    "mhe_time": configs[0],
                    "offline_file": data,
                    "interpol_factor": 2,
                    "source": processed_source[0],
                    "use_torque": True,
                    "save_results": True,
                    "track_emg": True,
                    "with_f_ext": True,
                    "f_ext_as_constraints": False,
                    "parameters_file_path": parameters_file_path,
                    "kin_data_to_track": "markers",
                    # "kin_data_to_track": "q",
                    "exp_freq": exp_freq[0],
                    "result_dir": result_dir,
                    "result_file_name": f"result_mhe_{trial}_{processed_source[0]}_optim_param_{c}.bio",
                    "solver_options": solver_options,
                    "weights": configure_weights(),
                    "frame_to_save": 0,
                    "save_all_frame": True,
                    "part": part,
                    "use_acados": True,
                    "use_optim_params": c,
                    # "emg_names": ["PECM",
                    #               "bic",
                    #               "tri",
                    #               "LAT",
                    #               'TRP1',
                    #               "DELT1",
                    #               'DELT2',
                    #               'DELT3']
                    "emg_names" : ["PectoralisMajorThorax",
                     "BIC",
                     "TRI",
                     "LatissimusDorsi",
                     'TrapeziusScapula_S',
                     #'TrapeziusClavicle',
                     "DeltoideusClavicle_A",
                     'DeltoideusScapula_M',
                      'DeltoideusScapula_P']
                }
                variables_dic = {"print_lvl": 1}  # print level 0 = no print, 1 = print information
                MHE = MuscleForceEstimator(configuration_dic)
                MHE.run_mhe(variables_dic, [])
            #break
