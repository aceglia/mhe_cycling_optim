from casadi import MX, Function, vertcat, sum1, sum2, reshape, nlpsol, repmat
import biorbd_casadi as biorbd_ca
import numpy as np
from optim_params.casadi_utils import Symbolics, MxVariables
from optim_params.enum import Parameters
from optim_params.identification_utils import (get_initial_values, map_activation, return_muscle_torque_function,
                                               get_cost_n_dependant, get_cost_to_map, get_initial_parameters,
                                               return_bounds, return_param_from_mapping)
from biosiglive import save
import time



class ParametersIdentifier:
    def __init__(self, params: list[Parameters]):
        self.model = None
        self.params_to_optim = [p.value for p in params]
        self.tau = None
        self.q = None
        self.q_dot = None
        self.emg = None
        self.is_data_loaded = False
        self.symbolics = None
        self.mx_variables = None

    def load_experimental_data(self, data:dict, prepare_data_function=None, **kwargs):
        self.tau = data['tau']
        self.q = data['q']
        self.q_dot = data['qdot']
        if "emg" in data.keys():
            self.emg = data['emg']
        self.is_data_loaded = True
        if prepare_data_function is not None:
            self.tau, self.q, self.q_dot, self.emg = prepare_data_function(self.tau, self.q, self.q_dot, self.emg, **kwargs)
    
    def _compute_symbolics(self):
        self.symbolics = Symbolics()
        self.symbolics.add("q", self.model.nbQ())
        self.symbolics.add("qdot", self.model.nbQ())
        self.symbolics.add("tau", self.tau.shape[0])
        if self.with_param:
            self.symbolics.add("p", self.n_p)
        self.symbolics.add("emg", self.emg.shape[0])
        if self.with_torque:
            self.symbolics.add("pas_tau", len(self.residual_torque_idx))
        self.symbolics.add("x", self.model.nbMuscles())

    def _compute_mx_variables(self, use_sx=False):
        self.mx_variables = MxVariables()
        self.mx_variables.add("q", self.q, sx=use_sx)
        self.mx_variables.add("qdot", self.q_dot, sx=use_sx)
        self.mx_variables.add("tau", self.tau, sx=use_sx)
        self.mx_variables.add("emg", self.emg, sx=use_sx)

    def _compute_mapped_cost_function(self, use_sx=False):
        j = get_cost_to_map(self.scaling_factor, self.symbolics,
                                  self.weights,
                                  p_mapping=self.p_mapping,
                                  with_torque=self.with_torque,
                                  muscle_track_idx=self.muscle_track_idx,
                                  muscle_casadi_function=self.ca_funct,
                                  with_param=self.with_param,
                                  passive_torque_idx=self.residual_torque_idx,
                                  tau_as_constraint=self.torque_as_constraint,
                                  ignore_dof=self.ignore_dof)

        symbolics_to_get = ["x", "q", "qdot", "tau", "emg"] + \
                           (["pas_tau"] if self.with_torque else []) + \
                           (["p"] if self.with_param else [])
        sym_list = self.symbolics.get(symbolics_to_get)
        J_func = Function("J1", sym_list, [j]).expand()
        J_mapped = J_func.map(self.ns, "thread", self.threads)
        self.symbolics.add("x_all", self.model.nbMuscles() * self.ns, sx=use_sx)
        self.symbolics.add("p_all", self.n_p, sx=use_sx)
        x_split = reshape(self.symbolics.x_all, self.model.nbMuscles(), self.ns)

        if self.with_torque:
            self.symbolics.add("pas_tau_all", len(self.residual_torque_idx) * self.ns, sx=use_sx)
            tau_split = reshape(self.symbolics.pas_tau_all, len(self.residual_torque_idx), self.ns)
            obj_1 = J_mapped(x_split, self.mx_variables.get("q"), self.mx_variables.get("qdot"),
                             self.mx_variables.get("tau"),
                             self.mx_variables.get("emg"), tau_split, repmat(self.symbolics.p_all, 1, self.ns))
        else:
            obj_1 = J_mapped(x_split, self.mx_variables.get("q"), self.mx_variables.get("qdot"),
                             self.mx_variables.get("tau"),
                             self.mx_variables.get("emg"), repmat(self.symbolics.p_all, 1, self.ns))
        obj_1 = sum2(obj_1)
        return obj_1

    def _compute_non_mapped_cost_function(self, model_param_init, l_norm_bounded=False):
        J_2, g = get_cost_n_dependant(self.symbolics.p, self.p_mapping,
                                      self.params_to_optim, self.scaling_factor, self.weights,
                                  use_ratio_tracking=False, param_init=model_param_init,
                                      bounds_l_norm=l_norm_bounded, muscle_len=self.all_muscle_len)
        J_2_func = Function("J2", [self.symbolics.p], [J_2]).expand()
        if l_norm_bounded:
            g_fun = Function("g", [self.symbolics.p], [g]).expand()
            self.g = sum1(g_fun(self.symbolics.p_all))
        else:
            self.g = None
        obj_2 = J_2_func(self.symbolics.p_all)
        return obj_2

    def initialize_problem(self,
                           biorbd_model_path:str,
                           p_mapping:list,
                           with_param=True,
                           with_residual_torques=True, torque_as_constraint=False, ignore_dof=None,
                           emg_names=None,
                           residual_torque_idx=None, scaling_factor=None, all_muscle_len=None, weights=None,
                           threads=6, l_norm_bounded=False, param_bounds=None, p_init=None, use_sx=True):
        self.g = None
        self.scaling_factor = scaling_factor
        self.weights = weights
        self.p_mapping = p_mapping
        self.with_param = with_param
        self.emg_names = emg_names
        self.with_torque = with_residual_torques
        self.torque_as_constraint = torque_as_constraint
        self.ignore_dof = ignore_dof
        self.all_muscle_len = all_muscle_len
        self.threads = threads
        self.model = biorbd_ca.Model(biorbd_model_path)
        self.l_norm_bounded = l_norm_bounded
        self.param_bounds = param_bounds
        self.p_init = p_init
        muscle_list = [name.to_string() for name in self.model.muscleNames()]
        muscle_track_idx = []
        self.tic = time.time()
        for i in range(len(emg_names)):
            muscle_track_idx.append([j for j in range(len(muscle_list)) if emg_names[i] in muscle_list[j]])
        self.muscle_track_idx = sum(muscle_track_idx, [])
        self.emg = map_activation(self.emg, emg_names=self.emg_names,
                                        muscle_track_idx=self.muscle_track_idx,
                                        model=self.model)

        self.n_p = sum([len(i[0]) for i in p_mapping]) if self.with_param else 0
        self.residual_torque_idx = [i for i in range(self.model.nbQ() - 10,
                                               self.model.nbGeneralizedTorque())] if residual_torque_idx is None else residual_torque_idx
        # define casadi variables
        self.ns = self.q.shape[1]
        self._compute_symbolics()
        self._compute_mx_variables(use_sx=use_sx)

        model_param_init, ratio_init = get_initial_parameters(self.model, self.params_to_optim)
        self.ca_funct = return_muscle_torque_function(self.model, self.symbolics, p_mapping,
                                      with_param=self.with_param,
                                      params_to_optim=self.params_to_optim,
                                      model_params_init=model_param_init,
                                                 ratio=ratio_init)

        obj_1 = self._compute_mapped_cost_function(use_sx=use_sx)
        obj_2 = self._compute_non_mapped_cost_function(model_param_init)
        total_obj = obj_1 + obj_2
        # total_obj /= 100
        self.total_obj = total_obj

    def solve(self, save_results=False, output_file=None, **kwargs):
        opts = {"ipopt": {"max_iter": 1000, "print_level": 5, "linear_solver": "ma57",
                          "hessian_approximation": "exact",
                          "acceptable_tol": 1e-2,
                          "tol": 1e-2
                          }}
        opts["ipopt"].update(**kwargs)
        x0, tau_0 = get_initial_values(self.model, self.residual_torque_idx, self.ns, self.muscle_track_idx, self.emg,
                                       self.scaling_factor)

        bounds_dic = return_bounds(self.model, self.scaling_factor, self.symbolics.p_all, self.ns,
                                   self.symbolics.x_all, self.symbolics.pas_tau_all, x0, tau_0, self.with_param,
                                    self.with_torque,
                                    self.p_init, self.params_to_optim, self.p_mapping, self.param_bounds,
                                   self.l_norm_bounded, tau_bounds=50
                                    )
        if self.torque_as_constraint or self.g is not None:
            nlp = {"x": bounds_dic["x"], "f": self.total_obj, "g": self.g}
            sol_nlp = nlpsol("sol", "ipopt", nlp, opts)
            solution = sol_nlp(x0=bounds_dic["x0"], lbx=bounds_dic["lbx"], ubx=bounds_dic["ubx"], lbg=bounds_dic["lbg"],
                               ubg=bounds_dic["ubg"])
        else:
            nlp = {"x": bounds_dic["x"], "f": self.total_obj,}
            sol_nlp = nlpsol("sol", "ipopt", nlp, opts)
            solution = sol_nlp(x0=bounds_dic["x0"], lbx=bounds_dic["lbx"], ubx=bounds_dic["ubx"])
        return self._dispatch_results(solution, sol_nlp, save_results, output_file)
    
    def _dispatch_results(self, solution, sol_nlp, save_results, output_file):
        all_states = solution["x"].toarray().squeeze()
        pas_tau_mat = np.zeros(self.tau.shape)
        if self.with_torque:
            residual_torques = all_states[-len(self.residual_torque_idx) * self.ns:].reshape(-1, len(self.residual_torque_idx)).T
            pas_tau_mat[self.residual_torque_idx, :] = residual_torques
        act = all_states[:self.model.nbMuscles() * self.ns].reshape(-1, self.model.nbMuscles()).T
        if self.with_torque:
            p = all_states[self.model.nbMuscles() * self.ns:-len(self.residual_torque_idx) * self.ns]
        else:
            p = all_states[self.model.nbMuscles() * self.ns:]
        p_list = return_param_from_mapping(self.p_mapping, p)
        solver_out = {"n_iter": sol_nlp.stats()["iter_count"], "status": sol_nlp.stats()["success"],
                      "return_status": sol_nlp.stats()["return_status"],
                      "nb_total_iter": len(sol_nlp.stats()["iterations"]["alpha_du"])}
        if save_results:
            self._save_results(act, pas_tau_mat, p_list,solver_out, output_file)
        return act, pas_tau_mat, p_list, solver_out

    def _save_results(self, act, pas_tau, p, solver_out, save_path):
        save({"a": act, "pas_tau": pas_tau, "p": p, "emg": self.emg, "q": self.q, "qdot": self.q_dot,
              "scaling_factor": self.scaling_factor,
              "p_mapping": self.p_mapping, "p_init": self.p_init, "solving_time": time.time() - self.tic,
              "optimized_params": self.params_to_optim,
              "tracked_torque": self.tau, "muscle_track_idx": self.muscle_track_idx,
              "param_bounds": self.param_bounds, "solver_out": solver_out}, save_path,
             safe=False,
             #add_data=True
             )