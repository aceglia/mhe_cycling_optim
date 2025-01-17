import biorbd
from casadi import MX, Function, vertcat, sum1, sum2, reshape, nlpsol, repmat
import biorbd_casadi as biorbd_ca
import numpy as np
from optim_params.casadi_utils import Symbolics, MxVariables, convert_to_casadi
from optim_params.enum import Parameters
from optim_params.identification_utils import (
    get_initial_values,
    map_activation,
    return_muscle_torque_function,
    get_cost_n_dependant,
    get_cost_to_map,
    get_initial_parameters,
    return_bounds,
    return_param_from_mapping,
    compute_muscle_joint_torque,
    apply_params,
    optimize_parameters_init,
    return_obj_function,
    _get_muscle_torque,
    return_muscle_length_function,
    _get_muscle_length,
)
from optim_params.plt_utils import (
    plot_joint_torques,
    plot_param,
    plot_muscle_activation,
    plot_muscle_force,
    plot_norm_length,
)
from biosiglive import save
import time


class ParametersIdentifier:
    def __init__(self, params: list[Parameters]):
        self.solving_time = None
        self.lm_optim_init = None
        self.model = None
        self.params_to_optim = [p.value for p in params]
        self.tau = None
        self.q = None
        self.q_dot = None
        self.emg = None
        self.is_data_loaded = False
        self.symbolics = None
        self.mx_variables = None

    def load_experimental_data(self, data: dict, prepare_data_function=None, **kwargs):
        self.tau = data["tau"]
        self.q = data["q"]
        self.q_dot = data["qdot"]
        if "emg" in data.keys():
            self.emg = data["emg"]
        self.is_data_loaded = True
        if prepare_data_function is not None:
            self.tau, self.q, self.q_dot, self.emg = prepare_data_function(
                self.tau, self.q, self.q_dot, self.emg, **kwargs
            )

    def _compute_symbolics(self):
        self.symbolics = Symbolics()
        self.symbolics.add("q", self.model.nbQ())
        self.symbolics.add("qdot", self.model.nbQ())
        self.symbolics.add("tau", self.tau.shape[0])
        self.symbolics.add("muscles_len", self.model.nbMuscles())
        self.symbolics.add("lm_init", self.model.nbMuscles())
        self.symbolics.__dict__["muscle_torque_all"] = None
        if self.add_muscle_torque_constraint:
            self.symbolics.add("muscle_torque", self.model.nbQ())
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
        self.mx_variables.add("MTU_len", self.MTU_len, sx=use_sx)
        self.mx_variables.add("lm_init", convert_to_casadi(self.lm_optim_init, to_array=True), sx=use_sx)

    def _compute_mapped_cost_function(self, model_param_init, use_sx=False):
        self.symbolics.add("lm_optim", self.model.nbMuscles())
        j, g = get_cost_to_map(
            self.scaling_factor,
            self.symbolics,
            self.weights,
            p_mapping=self.p_mapping,
            with_torque=self.with_torque,
            muscle_track_idx=self.muscle_track_idx,
            muscle_casadi_function=self.ca_funct,
            len_casadi_function=self.len_function,
            with_param=self.with_param,
            passive_torque_idx=self.residual_torque_idx,
            tau_as_constraint=self.torque_as_constraint,
            ignore_dof=self.ignore_dof,
            bounds_l_norm=self.l_norm_bounded,
            params_to_optim=self.params_to_optim,
            muscle_torque_as_constraint=self.add_muscle_torque_constraint,
        )

        symbolics_to_get = (
            ["x", "q", "qdot", "tau", "emg"]
            + (["pas_tau"] if self.with_torque else [])
            + (["p"] if self.with_param else [])
            + (["muscle_torque"] if self.add_muscle_torque_constraint else [])
            + (["lm_init"] if "lm_optim" in self.params_to_optim else [])
        )
        # (["muscles_len"] if "lm_optim" in self.params_to_optim else [])

        sym_list = self.symbolics.get(symbolics_to_get)
        J_func = Function("J1", sym_list, [j]).expand()
        # g = self.symbolics.pas_tau[5:9]
        # g_func = Function("g1", self.symbolics.get(["pas_tau"]), [g]).expand()
        if self.l_norm_bounded:
            g_fun = Function(
                "g", [self.symbolics.q, self.symbolics.p, self.symbolics.lm_init, self.symbolics.lm_optim], [g]
            ).expand()

        if self.add_muscle_torque_constraint:
            self.symbolics.add("muscle_tau_from_act", self.model.nbQ())
            # f = Function("msj", [x, q, qdot, p], [mus_tau_from_act])
            g = self.symbolics.muscle_tau_from_act - self.symbolics.muscle_torque
            g_func = Function("g1", self.symbolics.get(["muscle_torque", "muscle_tau_from_act"]), [g]).expand()
        # else:
        #     g_func = None
        self.symbolics.add("x_all", self.model.nbMuscles() * self.ns, sx=use_sx)
        self.symbolics.add("p_all", self.n_p, sx=use_sx)
        x_split = reshape(self.symbolics.x_all, self.model.nbMuscles(), self.ns)
        tau_split = None
        if self.with_torque:
            self.symbolics.add("pas_tau_all", len(self.residual_torque_idx) * self.ns, sx=use_sx)
            tau_split = reshape(self.symbolics.pas_tau_all, len(self.residual_torque_idx), self.ns)
        tau_muscle_split = None
        if self.add_muscle_torque_constraint:
            self.symbolics.add("muscle_torque_all", self.model.nbQ() * self.ns, sx=use_sx)
            tau_muscle_split = reshape(self.symbolics.muscle_torque_all, self.model.nbQ(), self.ns)

        if self.threads == 1:
            obj_1 = []
            g_1 = None if g is None else []
            p_lm_optim = None
            if "lm_optim" in self.params_to_optim:
                idx_lm_optim = self.params_to_optim.index("lm_optim")
                all_lm_optim = sum([len(p_map[0]) for p_map in self.p_mapping[:idx_lm_optim]])
                p_lm_optim = self.symbolics.p_all[all_lm_optim : all_lm_optim + len(self.p_mapping[idx_lm_optim][0])]
            for i in range(self.ns):
                tau_muscle_tmp = None if tau_muscle_split is None else tau_muscle_split[:, i]
                tau_tmp = None if tau_split is None else tau_split[:, i]
                # muscle_len = _get_muscle_length(self.mx_variables.q[:, i],
                #                                 self.symbolics.p_all, self.p_mapping, self.len_function,
                #                                 self.scaling_factor,
                #                                 self.with_param)
                obj_tmp = return_obj_function(
                    J_func,
                    x_split[:, i],
                    self.mx_variables.get("q")[:, i],
                    self.mx_variables.get("qdot")[:, i],
                    self.mx_variables.get("tau")[:, i],
                    self.mx_variables.get("emg")[:, i],
                    self.symbolics.p_all,
                    tau_tmp,
                    tau_muscle_tmp,
                    self.mx_variables.lm_init,
                )
                if self.l_norm_bounded:
                    g_tmp = g_fun(
                        self.mx_variables.get("q")[:, i], self.symbolics.p_all, self.mx_variables.lm_init, p_lm_optim
                    )
                    g_1 = vertcat(g_1, g_tmp)
                obj_1 = vertcat(obj_1, obj_tmp)

                if self.add_muscle_torque_constraint:
                    mus_tau_from_act = _get_muscle_torque(
                        x_split[:, i],
                        self.mx_variables.get("q")[:, i],
                        self.mx_variables.get("qdot")[:, i],
                        self.symbolics.p_all,
                        self.p_mapping,
                        self.ca_funct,
                        self.scaling_factor,
                        self.with_param,
                    )
                    g_1 = vertcat(g_1, g_func(tau_muscle_tmp, mus_tau_from_act))
            obj_1 = sum1(obj_1)
            # g_1 = sum1(g_1)
        else:
            J_mapped = J_func.map(self.ns, "thread", self.threads)
            if self.l_norm_bounded:
                idx_lm_optim = self.params_to_optim.index("lm_optim")
                all_lm_optim = sum([len(p_map[0]) for p_map in self.p_mapping[:idx_lm_optim]])

                g_fun = Function(
                    "g", [self.symbolics.muscles_len, self.symbolics.lm_init, self.symbolics.lm_optim], [g]
                ).expand()
                g_mapped = g_fun.map(self.ns, "thread", self.threads)
                p_lm_optim = self.symbolics.p_all[all_lm_optim : all_lm_optim + len(self.p_mapping[idx_lm_optim][0])]
                g = g_mapped(
                    self.mx_variables.muscles_len,
                    repmat(self.mx_variables.lm_init, 1, self.ns),
                    repmat(p_lm_optim, 1, self.ns),
                )
            obj_1 = return_obj_function(
                J_mapped,
                x_split,
                self.mx_variables.get("q"),
                self.mx_variables.get("qdot"),
                self.mx_variables.get("tau"),
                self.mx_variables.get("emg"),
                tau_split,
                repmat(self.symbolics.p_all, 1, self.ns),
                tau_muscle_split,
            )
            obj_1 = sum2(obj_1)
        return obj_1, g_1

    def _compute_non_mapped_cost_function(self, model_param_init, l_norm_bounded=False, use_sx=False):
        self.symbolics.add("p_non_map", self.n_p, use_sx)
        J_2, g = get_cost_n_dependant(
            self.symbolics.p_non_map,
            self.p_mapping,
            self.params_to_optim,
            self.scaling_factor,
            self.weights,
            mx_variables=self.mx_variables,
            use_ratio_tracking=False,
            param_init=model_param_init,
            bounds_l_norm=l_norm_bounded,
            use_sx=use_sx,
        )
        J_2_func = Function("J2", [self.symbolics.p_non_map], [J_2]).expand()
        # if l_norm_bounded:
        #     g_fun = Function("g", [self.symbolics.p_non_map], [g]).expand()
        #     self.g = sum1(g_fun(self.symbolics.p_all))
        # else:
        #     self.g = None
        obj_2 = J_2_func(self.symbolics.p_all)
        return obj_2

    def initialize_problem(
        self,
        biorbd_model_path: str,
        p_mapping: list,
        with_param=True,
        with_residual_torques=True,
        torque_as_constraint=False,
        ignore_dof=None,
        emg_names=None,
        residual_torque_idx=None,
        scaling_factor=None,
        all_muscle_len=None,
        weights=None,
        threads=1,
        l_norm_bounded=False,
        param_bounds=None,
        p_init=None,
        use_sx=True,
        add_muscle_torque_constraint=True,
    ):
        self.g = None
        self.scaling_factor = scaling_factor
        self.weights = weights
        self.p_mapping = p_mapping
        self.with_param = with_param
        self.emg_names = emg_names
        self.with_torque = with_residual_torques
        self.torque_as_constraint = torque_as_constraint
        self.ignore_dof = ignore_dof
        self.MTU_len = all_muscle_len
        self.threads = threads
        self.model = biorbd_ca.Model(biorbd_model_path)
        self.l_norm_bounded = l_norm_bounded
        self.add_muscle_torque_constraint = add_muscle_torque_constraint
        self.param_bounds = param_bounds
        self.p_init = p_init
        if self.p_init is None:
            self.p_init = [[1 for _ in range(len(p_mapping[p][0]))] for p in range(len(p_mapping))]
        muscle_list = [name.to_string() for name in self.model.muscleNames()]
        muscle_track_idx = []
        self.tic = time.time()
        for i in range(len(emg_names)):
            muscle_track_idx.append([j for j in range(len(muscle_list)) if emg_names[i] in muscle_list[j]])
        self.muscle_track_idx = sum(muscle_track_idx, [])
        self.emg = map_activation(
            self.emg, emg_names=self.emg_names, muscle_track_idx=self.muscle_track_idx, model=self.model
        )

        self.n_p = sum([len(i[0]) for i in p_mapping]) if self.with_param else 0
        self.residual_torque_idx = (
            [i for i in range(self.model.nbQ() - 10, self.model.nbGeneralizedTorque())]
            if residual_torque_idx is None
            else residual_torque_idx
        )
        # define casadi variables
        model_param_init, ratio_init = get_initial_parameters(self.model, self.params_to_optim)
        if "lm_optim" in self.params_to_optim:
            self.lm_optim_init = model_param_init[self.params_to_optim.index("lm_optim")]

        # lm_optim = np.array(convert_to_casadi(model_param_init[self.params_to_optim.index("lm_optim")], to_array=True)).reshape(-1, 1)
        # for m in range(self.model.nbMuscles()):
        #    norm_len = self.all_muscle_len[m, :] / (lm_optim[m, 0] * self.p_init[self.params_to_optim.index("lm_optim")][m])
        #    if 0.5 < norm_len.max() < 1.5 and 0.5 < norm_len.min() < 1.5:
        #        continue
        #    else:
        #        print(f"Muscle {m} has a norm length outside the range [0.5, 1.5] (min: {norm_len.min()}, max: {norm_len.max()}")
        #        self.p_init[self.params_to_optim.index("lm_optim")][m] = optimize_parameters_init(self.all_muscle_len[m, :], lm_optim[m, 0])
        #        print(f"New value for lm_optim[{m}] is {self.p_init[self.params_to_optim.index('lm_optim')][m]}")

        self.ns = self.q.shape[1]
        self._compute_symbolics()
        self._compute_mx_variables(use_sx=use_sx)
        self.ca_funct = return_muscle_torque_function(
            self.model,
            self.symbolics,
            p_mapping,
            with_param=self.with_param,
            params_to_optim=self.params_to_optim,
            model_params_init=model_param_init,
            ratio=ratio_init,
        )

        self.len_function = return_muscle_length_function(
            self.model,
            self.symbolics.q,
            p_init=self.symbolics.p,
            with_param=self.with_param,
            p_mapping=p_mapping,
            params_to_optim=self.params_to_optim,
            model_params_init=model_param_init,
            ratio=ratio_init,
        )
        obj_1, g = self._compute_mapped_cost_function(model_param_init, use_sx=use_sx)
        self.g = g
        # self.g = None
        obj_2 = self._compute_non_mapped_cost_function(model_param_init, self.l_norm_bounded, use_sx=use_sx)
        total_obj = obj_1 + obj_2

        if self.g is not None:
            self.g = reshape(self.g, (-1, 1))
            # self._compute_constraint_hessian()

        self.total_obj = total_obj
        # self._compute_objective_hessian()

    def _compute_objective_hessian(self):
        import casadi as ca

        size = [self.symbolics.p_all.shape[0], self.symbolics.x_all.shape[0]]
        x_tmp = ca.MX.ones(self.symbolics.x_all.shape[0]) * 0.5 * self.scaling_factor[0]
        p_tmp = ca.MX.ones(self.symbolics.p_all.shape[0])
        hess_params = ca.vertcat(self.symbolics.p_all, self.symbolics.x_all)
        hes_params_num = ca.vertcat(p_tmp, x_tmp)
        if self.with_torque:
            size.append(self.symbolics.pas_tau_all.shape[0])
            hess_params = ca.vertcat(hess_params, self.symbolics.pas_tau_all)
            pas_tau_tmp = ca.MX.ones(self.symbolics.pas_tau_all.shape[0]) * 0.5 * self.scaling_factor[2]
            hes_params_num = ca.vertcat(hes_params_num, pas_tau_tmp)
        if self.add_muscle_torque_constraint:
            size.append(self.symbolics.muscle_torque_all.shape[0])
            muscle_torque_tmp = ca.MX.ones(self.symbolics.muscle_torque_all.shape[0]) * self.scaling_factor[2]
            hes_params_num = ca.vertcat(hes_params_num, muscle_torque_tmp)
            hess_params = ca.vertcat(hess_params, self.symbolics.muscle_torque_all)
        jac_fct_f_all = ca.hessian(self.total_obj, hess_params)[0]
        jac_fct_f = ca.Function("jac_fct_f", [hess_params], [jac_fct_f_all])
        jac_func_num = ca.Function("pouet", [], [jac_fct_f(hes_params_num)])()["o0"].toarray().squeeze()
        eigen_values = np.linalg.eigvals(jac_func_num)
        ev_max = min(eigen_values)
        ev_min = max(eigen_values)
        if ev_min == 0:
            condition_number = "! Ev_min is 0"
        if ev_min != 0:
            condition_number = np.abs(ev_max) / np.abs(ev_min)
        convexity = "Possible"
        for ev in range(eigen_values.size):
            if eigen_values[ev] < 0:
                convexity = "False"
                print("Convexity is false")
                break
        max_hes = np.ndarray.max(jac_func_num)
        min_hes = np.ndarray.min(jac_func_num)
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        names = ["params", "x", "res_tau", "muscle_torque"]
        count = 0
        for i in range(len(size)):
            fig_obj = plt.figure(names[i])
            if names[i] != "params":
                jac_tmp = jac_func_num[count : count + size[i] // self.ns, count : count + size[i] // self.ns]
            else:
                jac_tmp = jac_func_num[count : count + size[i], count : count + size[i]]
            count += size[i]
            # fig_obj, axis_obj = plt.subplots(1, 1)
            jac_tmp[~(jac_tmp != 0).astype(bool)] = np.nan
            current_cmap3 = mcm.get_cmap("seismic")
            current_cmap3.set_bad(color="k")
            norm = mcolors.TwoSlopeNorm(vmin=min_hes - 0.01, vmax=max_hes + 0.01, vcenter=0)
            im3 = plt.imshow(jac_tmp, cmap=current_cmap3, norm=norm)
            cbar_ax3 = fig_obj.add_axes([0.02, 0.4, 0.015, 0.3])
            fig_obj.colorbar(im3, cax=cbar_ax3)
        # plt.show()

    def _compute_constraint_hessian(self):
        import casadi as ca

        size = [self.symbolics.p_all.shape[0], self.symbolics.x_all.shape[0], self.symbolics.muscle_torque_all.shape[0]]
        x_tmp = ca.MX.ones(self.symbolics.x_all.shape[0]) * 0.9 * self.scaling_factor[0]
        p_tmp = ca.MX.ones(self.symbolics.p_all.shape[0])
        muscle_torque_tmp = ca.MX.ones(self.symbolics.muscle_torque_all.shape[0]) * self.scaling_factor[2] * 10
        hess_params = ca.vertcat(self.symbolics.p_all, self.symbolics.x_all, self.symbolics.muscle_torque_all)
        hes_params_num = ca.vertcat(p_tmp, x_tmp, muscle_torque_tmp)
        # g = ca.sum1(self.g)
        jac_fct_g_all = ca.jacobian(self.g, hess_params)
        jac_fct_g = ca.Function("jac_fct_f", [hess_params], [jac_fct_g_all])
        jac_func_num = ca.Function("convertion", [], [jac_fct_g(hes_params_num)])()["o0"].toarray().squeeze()
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        names = ["g"]
        count = 0
        max_hes = np.ndarray.max(jac_func_num)
        min_hes = np.ndarray.min(jac_func_num)
        for i in range(len(size)):
            fig_obj = plt.figure(names[0])
            jac_tmp = jac_func_num[:, :]
            # fig_obj, axis_obj = plt.subplots(1, 1)
            jac_tmp[~(jac_tmp != 0).astype(bool)] = np.nan
            current_cmap3 = mcm.get_cmap("seismic")
            current_cmap3.set_bad(color="k")
            norm = mcolors.TwoSlopeNorm(vmin=min_hes - 0.01, vmax=max_hes + 0.01, vcenter=0)
            im3 = plt.imshow(jac_tmp, cmap=current_cmap3, norm=norm)
            cbar_ax3 = fig_obj.add_axes([0.02, 0.4, 0.015, 0.3])
            fig_obj.colorbar(im3, cax=cbar_ax3)

    def solve(
        self,
        save_results=False,
        output_file=None,
        plot=False,
        objective_scale_factor=1,
        batch_number=0,
        cycle_number=-1,
        **kwargs
    ):
        self.cycle_number, self.batch_number = cycle_number, batch_number
        opts = {
            "ipopt": {
                "max_iter": 1000,
                "print_level": 5,  # , "linear_solver": "ma57",
                "linear_solver": "ma57",
                "hessian_approximation": "exact",
                # "acceptable_tol": 1e-6,
                # "tol": 1e-3
            }
        }
        opts["ipopt"].update(**kwargs)
        self.total_obj /= objective_scale_factor
        x0, tau_0 = get_initial_values(
            self.model, self.residual_torque_idx, self.ns, self.muscle_track_idx, self.emg, self.scaling_factor
        )
        # if self.l_norm_bounded:
        #     normalized_l = (self.all_muscle_len /
        #                     np.repeat(
        #                         np.array(convert_to_casadi(self.lm_optim_init, to_array=True)).reshape(-1, 1),
        #                         self.all_muscle_len.shape[1], axis=1))
        #     bounds = [0, 2]
        #     init_params_values = []
        #     for l, l_norm in enumerate(normalized_l):
        #         min = l_norm.min()
        #         max = l_norm.max()
        #         if bounds[0] < min < bounds[1] and bounds[0] < max < bounds[1]:
        #             init_params_values.append(1)
        #         else:
        #             init_params_values.append(-1)

        bounds_dic = return_bounds(
            self.model,
            self.scaling_factor,
            self.symbolics.p_all,
            self.ns,
            self.symbolics.x_all,
            self.symbolics.pas_tau_all,
            self.symbolics.muscle_torque_all,
            x0,
            tau_0,
            self.with_param,
            self.with_torque,
            self.p_init,
            self.params_to_optim,
            self.p_mapping,
            self.param_bounds,
            self.l_norm_bounded,
            self.g,
            tau_bounds=20,
        )
        if self.torque_as_constraint or self.g is not None:
            nlp = {"x": bounds_dic["x"], "f": self.total_obj, "g": self.g}
            sol_nlp = nlpsol("sol", "ipopt", nlp, opts)
            solution = sol_nlp(
                x0=bounds_dic["x0"],
                lbx=bounds_dic["lbx"],
                ubx=bounds_dic["ubx"],
                lbg=bounds_dic["lbg"],
                ubg=bounds_dic["ubg"],
            )
        else:
            nlp = {"x": bounds_dic["x"], "f": self.total_obj}
            sol_nlp = nlpsol("sol", "ipopt", nlp, opts)
            solution = sol_nlp(x0=bounds_dic["x0"], lbx=bounds_dic["lbx"], ubx=bounds_dic["ubx"])
        self.solving_time = time.time() - self.tic
        print("Time to build and solve: ", self.solving_time)
        return self._dispatch_results(solution, sol_nlp, save_results, output_file, plot)

    def _dispatch_results(self, solution, sol_nlp, save_results, output_file, plot):
        muscle_torque = None
        all_states = solution["x"].toarray().squeeze()
        pas_tau_mat = np.zeros(self.tau.shape)
        if self.with_torque:
            residual_torques = (
                all_states[
                    self.model.nbMuscles() * self.ns
                    + self.n_p : self.model.nbMuscles() * self.ns
                    + self.n_p
                    + len(self.residual_torque_idx) * self.ns
                ]
                .reshape(-1, len(self.residual_torque_idx))
                .T
            )
            pas_tau_mat[self.residual_torque_idx, :] = residual_torques
            pas_tau_mat /= self.scaling_factor[2]
        act = all_states[: self.model.nbMuscles() * self.ns].reshape(-1, self.model.nbMuscles()).T
        act /= self.scaling_factor[0]
        p = all_states[self.model.nbMuscles() * self.ns : self.model.nbMuscles() * self.ns + self.n_p]

        p_list = np.array(return_param_from_mapping(self.p_mapping, p))
        for p_idx in range(len(p_list)):
            p_list[p_idx] /= self.scaling_factor[1][p_idx]
        if self.add_muscle_torque_constraint:
            muscle_torque = all_states[-self.model.nbQ() * self.ns :].reshape(-1, self.model.nbQ()).T
            muscle_torque /= self.scaling_factor[2]
        solver_out = {
            "n_iter": sol_nlp.stats()["iter_count"],
            "status": sol_nlp.stats()["success"],
            "return_status": sol_nlp.stats()["return_status"],
            "nb_total_iter": len(sol_nlp.stats()["iterations"]["alpha_du"]),
        }
        if save_results:
            self._save_results(act, pas_tau_mat, p_list, solver_out, output_file)
        if plot:
            self._plot(act, pas_tau_mat, p_list, muscle_torque)
        return act, pas_tau_mat, p_list, muscle_torque, solver_out

    def _save_results(self, act, pas_tau, p, solver_out, save_path):
        save(
            {
                "a": act,
                "pas_tau": pas_tau,
                "p": p,
                "emg": self.emg,
                "q": self.q,
                "qdot": self.q_dot,
                "scaling_factor": self.scaling_factor,
                "p_mapping": self.p_mapping,
                "p_init": self.p_init,
                "solving_time": self.solving_time,
                "optimized_params": self.params_to_optim,
                "tracked_torque": self.tau,
                "muscle_track_idx": self.muscle_track_idx,
                "param_bounds": self.param_bounds,
                "solver_out": solver_out,
                "cycle_number": self.cycle_number,
                "batch_number": self.batch_number,
                "weights": self.weights,
            },
            save_path,
            add_data=True,
        )

    def _plot(self, act, pas_tau_mat, p_list, muscle_torque):
        import matplotlib.pyplot as plt

        eigen_model = biorbd.Model(self.model.path().absolutePath().to_string())
        model_param_init, ratio_init = get_initial_parameters(eigen_model, self.params_to_optim, use_mx=False)
        model_updated = apply_params(
            eigen_model, p_list, self.params_to_optim, model_param_init, ratio=ratio_init, with_casadi=False
        )
        mjt = np.zeros((self.model.nbQ(), self.ns))
        for i in range(self.ns):
            mjt[:, i] = compute_muscle_joint_torque(
                model_updated, act[:, i], self.q[:, i], self.q_dot[:, i], with_param=False, to_mx=False
            )

        plot_param(
            p_list, [name.to_string() for name in eigen_model.muscleNames()], self.params_to_optim, self.param_bounds, 1
        )
        plot_joint_torques(mjt, self.tau, pas_tau_mat, muscle_torque)
        plot_muscle_activation(
            act, self.emg, [name.to_string() for name in eigen_model.muscleNames()], self.muscle_track_idx, self.q
        )
        plot_muscle_force(
            eigen_model,
            act,
            self.q,
            self.q_dot,
            [name.to_string() for name in eigen_model.muscleNames()],
        )

        plot_norm_length(eigen_model, self.q)
        plt.show()
