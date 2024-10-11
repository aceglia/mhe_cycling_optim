from optim_params.ocp_utils import prepare_ocp, get_update_function, get_solver_options
from bioptim import Solver, SolutionMerge, BiorbdModel
import numpy as np
from biosiglive import save


class TorqueEstimator:
    def __init__(self):
        self.torque_computed = None
        self.ocp_time = None
        self.kin_init = None
        self.bio_model = None
        self.model_path = None
        self.from_direct_dynamics = None
        self.ocp_initialized = None
        self.with_external_loads = False
        self.use_residuals = False
        self.model = None
        self.from_inverse_dynamics = False
        self.use_mhe = False
        self.track_previous = False
        self.n_threads = 6
        self.is_data_loaded = False
        self.f_ext = None
        self.q_init = None
        self.n_shooting = None
        self.final_time = None
        self.markers_target = None
        self.q_dot_init = None
        self.ocp = None

        # OCP variables
        self.q_ocp, self.q_dot_ocp, self.tau_ocp, self.f_ext_ocp = None, None, None, None

        # ID variables
        self.q_id, self.q_dot_id, self.q_ddot_id, self.tau_id = None, None, None, None

    def init_experimental_data(self, data_dict: dict):
        """
        Initialize experimental data. Providen dictionary should contain the following keys:
        - markers_target: list of markers positions
        - q_init: initial position from inverse kinematics
        - q_dot_init: initial velocity from inverse kinematics
        - f_ext: external forces (optional)
        :param data_dict:
        :return:
        """
        self.is_data_loaded = True
        for key, value in data_dict.items():
            setattr(self, key, value)
        self.kin_init = np.concatenate((self.q_init, self.q_dot_init), axis=0)


    def init_ocp(self, biorbd_model_path, ocp_time, n_shooting,
                 with_external_loads=False, use_mhe = True, track_previous=False, n_threads=6, weights=None):
        self.use_mhe = use_mhe
        self.track_previous = track_previous
        self.n_threads = n_threads
        self.n_shooting = n_shooting
        self.ocp_time = ocp_time
        self.with_external_loads = with_external_loads
        self.ocp, self.bio_model = prepare_ocp(biorbd_model_path=biorbd_model_path,
                          final_time=self.ocp_time,
                          n_shooting=self.n_shooting,
                          use_sx=use_mhe,
                          n_threads=6,
                          kin_init=self.kin_init,
                          f_ext=self.f_ext,
                          target=self.markers_target[..., :self.n_shooting + 1],
                          with_f_ext=with_external_loads,
                          mhe=use_mhe,
                          track_previous=track_previous,
                          weights=weights
                          )
        self.ocp_initialized = True

    def compute_torque(self, from_direct_dynamics=False, from_inverse_dynamics=False, use_residuals=False,
                   with_external_loads=False, model_path=None, output_path=None, save_data=False, adapt_size_to_ocp=True):
        if not self.is_data_loaded:
            raise ValueError("Experimental data not loaded")
        if not from_direct_dynamics and not from_inverse_dynamics:
            raise ValueError("Please specify if you want to use direct and/or inverse dynamics")
        self.from_inverse_dynamics = from_inverse_dynamics
        self.from_direct_dynamics = from_direct_dynamics
        self.use_residuals = use_residuals
        self.with_external_loads = with_external_loads if self.with_external_loads is None else self.with_external_loads
        self.model_path = model_path
        if self.from_direct_dynamics:
            if not self.ocp_initialized:
                raise ValueError("OCP not initialized. Pleas call init_ocp() first.")
            self._get_torque_from_forward_dynamics()
        if self.from_inverse_dynamics:
            if self.bio_model is None:
                self.bio_model = BiorbdModel(self.model_path)
            self._get_torque_from_inverse_dynamics()
        self.torque_computed = True

        if save_data:
            self._save_data(output_path, adapt_size_to_ocp)

    def get_computed_torque(self, from_direct_dynamics=True, from_inverse_dynamics=False):
        return self._get_data(from_direct_dynamics=from_direct_dynamics,
                              from_inverse_dynamics=from_inverse_dynamics, key="torque")

    def get_computed_kinematics(self, from_direct_dynamics=True, from_inverse_dynamics=False):
        return self._get_data(from_direct_dynamics=from_direct_dynamics,
                              from_inverse_dynamics=from_inverse_dynamics, key="kin")

    def _get_data(self, from_direct_dynamics=True, from_inverse_dynamics=False, key=None):
        if from_direct_dynamics and from_inverse_dynamics:
            raise ValueError("Please specify only one of the two options")
        if from_direct_dynamics and not self.from_direct_dynamics:
            raise ValueError("Please call compute_torque() with from_direct_dynamics=True first")
        if from_inverse_dynamics and not self.from_inverse_dynamics:
            raise ValueError("Please call compute_torque() with from_inverse_dynamics=True first")
        if not self.torque_computed:
            raise ValueError("Torque not computed. Please call compute_torque() first.")
        if key == "kin" and from_direct_dynamics:
            return self.q_ocp, self.q_dot_ocp
        elif key == "torque" and from_direct_dynamics:
            return self.tau_ocp
        elif key == "kin" and from_inverse_dynamics:
            return self.q_id, self.q_dot_id, self.q_ddot_id
        elif key == "torque" and from_inverse_dynamics:
            return self.tau_id
        else:
            raise ValueError("Please specify a valid key")


    def _get_torque_from_inverse_dynamics(self):
        raise NotImplementedError("Inverse dynamics torque computation not implemented yet")

    def _get_torque_from_forward_dynamics(self):
        if self.use_mhe:
            sol = self.ocp.solve(get_update_function(self.markers_target, self.f_ext, self.with_external_loads,
                                                     self.track_previous, self.kin_init, self.n_shooting,
                                                     self.bio_model, self.ocp)
                            , **get_solver_options(Solver.ACADOS()))

        else:
            solver = Solver.IPOPT()
            solver.set_linear_solver("ma57")
            solver.set_hessian_approximation("exact")
            solver.set_tol(1e-5)
            solver.set_maximum_iterations(1000)
            sol = self.ocp.solve(solver=solver)

        merged_states = sol.decision_states(to_merge=SolutionMerge.NODES)
        merged_controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        self.q_ocp = merged_states["q"]
        self.q_dot_ocp = merged_states["q_dot"]
        self.tau_ocp = merged_controls["tau"]
        if self.with_external_loads:
            self.f_ext_ocp = merged_controls["f_ext"]

    def _save_data(self, output_path, adapt_size=True):
        final_data_to_save = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, np.ndarray) and (adapt_size and "ocp" not in key and self.use_mhe):
                final_data_to_save[key] = value[..., : - (self.n_shooting + 1)]
            else:
                final_data_to_save[key] = value
        save(final_data_to_save, output_path)




