from casadi import MX, Function, vertcat, sum1, sum2, reshape, nlpsol
import biorbd_casadi as biorbd_ca
import numpy as np
from casadi_utils import Symbolics, MxVariables
from identification_utils import (get_initial_values, map_activation, return_muscle_torque_function,
                                  get_cost_n_dependant)


class ParametersIdentifier:
    def __init__(self, params):
        self.model = None
        self.params = params
        self.tau = None
        self.q = None
        self.q_dot = None
        self.emg = None
        self.is_data_loaded = False
        self.symbolics = None
        self.mx_variables = None

    def load_experimental_data(self, data:dict, prepare_data_function=None, **kwargs):
        self.tau = data['reference_torque']
        self.q = data['q']
        self.q_dot = data['q_dot']
        if "emg" in data.keys():
            self.emg = data['emg']
        self.is_data_loaded = True
        if prepare_data_function is not None:
            self.tau, self.q, self.q_dot, self.emg = prepare_data_function(self.tau, self.q, self.q_dot, self.emg, **kwargs)
    
    def _compute_symbolics(self, with_param=True, with_torque=True, passive_torque_idx=None):
        self.symbolics = Symbolics()
        self.symbolics.add_symbolics("q", self.model.nbQ())
        self.symbolics.add_symbolics("qdot", self.model.nbQ())
        self.symbolics.add_symbolics("tau", self.tau.shape[0])
        if with_param:
            self.symbolics.add_symbolics("p", self.params.shape[0])
        self.symbolics.add_symbolics("emg", self.emg.shape[0])
        if with_torque:
            self.symbolics.add_symbolics("pas_tau", len(passive_torque_idx))

    def _compute_mx_variables(self):
        self.mx_variables = MxVariables()
        self.mx_variables.add_variable("q", self.model.nbQ())
        self.mx_variables.add_variable("qdot", self.model.nbQ())
        self.mx_variables.add_variable("tau", self.tau.shape[0])
        self.mx_variables.add_variable("emg", self.emg.shape[0])

    def run_identification(self,
                           params_to_optim:list,
                           biorbd_model_path:str,
                           p_mapping:list,
                           muscle_list:list, with_param=True,
                           with_torque=True, torque_as_constraint=False, ignore_dof=None,
                           emg_proc=None, emg_names=None, muscle_track_idx=None,
                           passive_torque_idx=None, scaling_factor=1, len_fct=None, weights=None, **kwargs):
        self.model = biorbd_ca.Model(biorbd_model_path)
        self.emg = map_activation(self.emg, emg_names=emg_names,
                                        muscle_track_idx=muscle_track_idx,
                                        model=self.model)

        n_p = len(sum([i[0] for i in p_mapping], []))
        passive_torque_idx = [i for i in range(self.model.nbQ() - 10,
                                               self.model.nbGeneralizedTorque())] if passive_torque_idx is None else passive_torque_idx
        # define casadi variables
        ns = self.q.shape[1]
        self._compute_symbolics(with_param=with_param, with_torque=with_torque, passive_torque_idx=passive_torque_idx)
        self._compute_mx_variables()

        # experimental data
        l_norm_bounded = False
        model_param_init, ratio_init = get_initial_values(self.model, params_to_optim)
        ca_funct = return_muscle_torque_function(self.model, self.symbolics, p_mapping,
                                      with_param=with_param,
                                      params_to_optim=params_to_optim,
                                      model_params_init=model_param_init,
                                                 ratio=ratio_init)

        j = get_cost_n_dependant(scaling_factor, self.symbolics,
                                  weights,
                                  p_mapping=p_mapping,
                                  with_torque=with_torque,
                                  muscle_track_idx=muscle_track_idx,
                                  muscle_casadi_function=ca_funct,
                                  with_param=with_param,
                                  passive_torque_idx=passive_torque_idx,
                                  tau_as_constraint=torque_as_constraint,
                                  ignore_dof=ignore_dof)

        symlist = [x_sym, q_sym, qdot_sym, tau_sym, emg_sym]
        if with_torque:
            symlist.append(pas_tau_sym)
        if with_param:
            symlist.append(p_sym)
        J_func = Function("J1", symlist, [j]).expand()
        J_mapped = J_func.map(ns, "thread", 4)
        # obj_1 = J_func(x_sym, q, qdot, tau, p_sym, pas_tau_sym, emg)
        x_all = MX.sym("x_all", model.nbMuscles() * ns)
        x_split = reshape(x_all, model.nbMuscles(), ns)
        p_all = MX.sym("p_all", n_p)

        if with_torque:
            pas_tau_all = MX.sym("pas_tau_all", len(passive_torque_idx) * ns)
            tau_split = reshape(pas_tau_all, len(passive_torque_idx), ns)
            obj_1 = J_mapped(x_split, q, qdot, tau, emg, tau_split, repmat(p_all, 1, ns))
        else:
            pas_tau_all = None
            obj_1 = J_mapped(x_split, q, qdot, tau, emg, repmat(p_all, 1, ns))
        obj_1 = sum2(obj_1)

        p_sym_2 = MX.sym("p_sym_2", n_p)
        J_2, g = _add_params_to_J(p_sym_2, p_mapping, params_to_optim, scaling_factor, use_ratio_tracking, weights,
                                  param_init=model_param_init,
                                  ratio=ratio_init, bounds_l_norm=l_norm_bounded, muscle_len=all_muscle_len)
        J_2_func = Function("J2", [p_sym_2], [J_2]).expand()
        if l_norm_bounded:
            g_fun = Function("g", [p_sym_2], [g]).expand()
            g = sum1(g_fun(p_all))
        obj_2 = sum1(J_2_func(p_all))
        total_obj = obj_1 + obj_2
        total_obj /= 100

        x0, tau_0 = _get_initial_values(model, passive_torque_idx, ns, muscle_track_idx, emg, scaling_factor)

        bounds_dic = _return_bounds(model, scaling_factor, p_all, ns, tau, x_all, pas_tau_all, x0, tau_0, with_param,
                                    with_torque,
                                    p_init, params_to_optim, p_mapping, param_bounds, l_norm_bounded,
                                    torque_as_constraint, dynamics_as_constraint)
        # bounds_dic["x0"][0:ns * model.nbMuscles()] = x0
        opts = {"ipopt": {"max_iter": 1000, "print_level": 5, "linear_solver": "ma57",
                          "hessian_approximation": "exact",
                          "acceptable_tol": 1e-2,
                          "tol": 1e-2,
                          # "nlp_scaling_method": None,
                          # "linear_system_scaling": None,
                          # "fast_step_computation": "yes"
                          }}
        # parallel.set_num_threads(4)
        if torque_as_constraint or g is not None:
            nlp = {"x": bounds_dic["x"], "f": total_obj, "g": g}
            sol_nlp = nlpsol("sol", "ipopt", nlp, opts)
            solution = sol_nlp(x0=bounds_dic["x0"], lbx=bounds_dic["lbx"], ubx=bounds_dic["ubx"], lbg=bounds_dic["lbg"],
                               ubg=bounds_dic["ubg"])
        else:
            nlp = {"x": bounds_dic["x"], "f": total_obj}
            sol_nlp = nlpsol("sol", "ipopt", nlp, opts)
            solution = sol_nlp(x0=bounds_dic["x0"], lbx=bounds_dic["lbx"], ubx=bounds_dic["ubx"])
        # if with_param:
        #     print(solution["x"][model.nbMuscles() * (ns):-tau.shape[0]] / scaling_factor[1] + 1)
        #     print(solution["x"][-tau.shape[0]:] / scaling_factor[2])
        act = np.zeros((model.nbMuscles(), ns))
        pas_tau_mat = np.zeros(tau.shape)
        if with_torque:
            sol_pas_tau = solution["x"][-len(passive_torque_idx) * ns:].toarray().squeeze()
        for j in range(ns):
            act[:, j] = np.array(solution["x"][
                                 j * model.nbMuscles(): j * model.nbMuscles() + model.nbMuscles()].toarray().squeeze()) / \
                        scaling_factor[0]
            if with_torque:
                count = 0
                for k in passive_torque_idx:
                    pas_tau_mat[k, j] = (np.array(
                        sol_pas_tau[j * len(passive_torque_idx): (j + 1) * len(passive_torque_idx)]) /
                                         scaling_factor[2])[count]
                    count += 1
        n_passive_tau = len(passive_torque_idx)
        to_add = 0
        if with_torque:
            p = np.array(solution["x"][model.nbMuscles() * ns:-n_passive_tau * ns]).squeeze()

        else:
            p = np.array(solution["x"][model.nbMuscles() * ns:]).squeeze()
            pas_tau_mat = None
        p_tmp = None
        count = 0
        for p_idx, param in enumerate(params_to_optim):
            n_p = len(p_mapping[p_idx][0])
            p_tmp = p[count:count + n_p] / scaling_factor[1][p_idx] if p_tmp is None else vertcat(p_tmp,
                                                                                                     p[
                                                                                                     count:count + n_p] /
                                                                                                     scaling_factor[1][
                                                                                                         p_idx])
            count += n_p
        p_list = _return_param_from_mapping(p_mapping, p_tmp)
        solver_out = {"n_iter": sol_nlp.stats()["iter_count"], "status": sol_nlp.stats()["success"],
                      "return_status": sol_nlp.stats()["return_status"]}
        return act, pas_tau_mat, p_list, emg, solver_out
    
