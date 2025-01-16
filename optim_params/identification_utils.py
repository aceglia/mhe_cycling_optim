import numpy as np
from casadi import Function, vertcat, MX
import casadi as ca
from scipy.interpolate import interp1d
import random
import itertools


def get_all_muscle_len(model, q):
    mus_list = np.zeros((model.nbMuscles(), q.shape[1]))
    for i in range(q.shape[1]):
        for m in range(model.nbMuscles()):
            mus_list[m, i] = model.muscle(m).length(model, q[:, i])
    return mus_list


def generate_random_idx(n_cycles, batch, n_data):
    random.seed(1000)
    combinations = list(itertools.combinations(list(range(1, int(n_data-1))), n_cycles))
    random_idx = random.sample(range(0, len(combinations)), batch)
    return [list(combinations[i]) for i in random_idx]

def _get_characteristics(muscle, name, use_mx=True):
    if name == "f_iso":
        return muscle.characteristics().forceIsoMax().to_mx() if use_mx else muscle.characteristics().forceIsoMax()
    elif name == "lm_optim":
        return muscle.characteristics().optimalLength().to_mx() if use_mx else muscle.characteristics().optimalLength()
    elif name == "lt_slack":
        return muscle.characteristics().tendonSlackLength().to_mx() if use_mx else muscle.characteristics().tendonSlackLength()


def get_initial_parameters(model, params_to_optim, use_mx=True):
    model_param_init = []
    count = 0
    ratio = []
    for p in params_to_optim:
        params_tmp = []
        for m in range(model.nbMuscles()):
            params_tmp.append(_get_characteristics(model.muscle(m), p, use_mx=use_mx))
            if count == 0:
                ratio.append(_get_characteristics(model.muscle(m), "lt_slack", use_mx=use_mx) / _get_characteristics(model.muscle(m), "lm_optim", use_mx=use_mx))
        count += 1
        model_param_init.append(params_tmp)
    return model_param_init, ratio


def map_activation(emg_proc, muscle_track_idx, model, emg_names):
    act = np.zeros((len(muscle_track_idx), int(emg_proc.shape[1])))
    init_count = 0
    for j, name in enumerate(emg_names):
        count = 0
        for i in range(model.nbMuscles()):
            if name in model.muscleNames()[i].to_string():
                count += 1
        act[list(range(init_count, init_count + count)), :] = emg_proc[j, :]
        init_count += count
    return act


def compute_muscle_joint_torque(model, activations, q, qdot,
                            p_init=None,
                                p_mapping=None,
                                params_to_optim=None, model_params_init=None, ratio=None,
                                with_param=True,
                                to_mx=True,
                            ):
    if with_param:
        param_list = return_param_from_mapping(p_mapping, p_init)
        model = apply_params(model, param_list, params_to_optim, model_params_init, ratio=ratio, with_casadi=to_mx)
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(activations[k])
    updated_model = model.UpdateKinematicsCustom(q, qdot)
    mjt = model.muscularJointTorque(muscles_states, q, qdot)
    mjt = mjt.to_mx() if to_mx else mjt.to_array()
    return mjt


def return_muscle_torque_function(model, symbolics,
                   p_mapping,
                   with_param=True,
                   params_to_optim=None,
                  model_params_init=None,
                   ratio=None):
    mjt = compute_muscle_joint_torque(model, symbolics.x, symbolics.q, symbolics.qdot, symbolics.p, p_mapping,
                                      params_to_optim, model_params_init, ratio, with_param=with_param)
    inputs = [symbolics.x, symbolics.q, symbolics.qdot]
    if with_param:
        inputs.append(symbolics.p)
    mjt_func = Function("mjt_func", inputs, [mjt]).expand()
    return mjt_func


def len_fct(model, q, with_param=True, p_mapping=None, p_init=None, params_to_optim=None, model_params_init=None, ratio=None, to_mx=True):
    if with_param:
        param_list = return_param_from_mapping(p_mapping, p_init)
        model = apply_params(model, param_list, params_to_optim, model_params_init, ratio=ratio, with_casadi=to_mx)
    mus_list = MX.zeros(model.nbMuscles(), 1)
    for i in range(model.nbMuscles()):
        mus_list[i] = model.muscle(i).length(model, q).to_mx()
    return mus_list

def return_muscle_length_function(model, q, with_param=True, p_mapping=None, p_init=None, params_to_optim=None, model_params_init=None, ratio=None, to_mx=True):
    muscle_len = len_fct(model, q, with_param, p_mapping, p_init, params_to_optim, model_params_init, ratio, to_mx)
    cas_fct = ca.Function("len_fct", [q, p_init], [muscle_len]).expand()
    return cas_fct


def get_cost_to_map(scaling_factor, symbolics, weights,
                           p_mapping=None,
                           with_torque=True,
                           muscle_track_idx=None,
                           muscle_casadi_function=None,
                           len_casadi_function=None,
                           with_param=True,
                           passive_torque_idx=None,
                           tau_as_constraint=False,
                           ignore_dof = None,
                            bounds_l_norm=False,
                            params_to_optim=None,
                            muscle_torque_as_constraint=False,
                    ):
    g = None
    j = 0
    x, q, qdot = symbolics.x, symbolics.q, symbolics.qdot
    p, act, tau, pas_tau = symbolics.p, symbolics.emg, symbolics.tau, symbolics.pas_tau

    if with_torque:
        torque_weights = np.array([weights["min_pas_torque"] for _ in range(pas_tau.shape[0])])
        for tau_idx in range(pas_tau.shape[0]):
            factor = 0 if tau_idx in [3] else 1
            factor = 0.1 if tau_idx in [9] else factor
           #if tau_idx in [5,6,7,8]:
           #    continue
            j += factor / torque_weights[tau_idx] * (pas_tau[tau_idx] ** 2)

    if "lm_optim" in params_to_optim:
        # j += weights["bound_lnorm"] * (symbolics.muscles_len / (
        #             symbolics.lm_init * (symbolics.lm_optim / scaling_factor[1][params_to_optim.index("lm_optim")]))) ** 2
        # idx_lm_optim = params_to_optim.index("lm_optim")
        # all_lm_optim = sum([len(p_map[0]) for p_map in p_mapping[:idx_lm_optim]])
        # p_lm_optim = p[all_lm_optim:all_lm_optim + len(p_mapping[idx_lm_optim][0])]
        muscle_len = _get_muscle_length(q, p, p_mapping, len_casadi_function, scaling_factor, with_param)
        norm_len = muscle_len / (symbolics.lm_init * (symbolics.lm_optim / scaling_factor[1][params_to_optim.index("lm_optim")]))

        if bounds_l_norm:
            g = norm_len
        # for i in range(symbolics.lm_optim.shape[0]):
        #     j += weights["bound_lnorm"] * norm_len[i, :]
        # for i in range(symbolics.lm_optim.shape[0]):
        # j += weights["bound_lnorm"] * (15000000 * ca.exp(200 - 500 * norm_len))
        else:
            j += weights["bound_lnorm"] * ca.exp(40 - 80 * norm_len)
        # j += weights["bound_lnorm"] * (15000000 * ca.exp(-50 * (norm_len)) ** 2)
           #J_params += (10000 * ca.sum2(norm_len[i, :] - 1)) **2
           # j += ca.sum2((1-ca.tanh(1000*(norm_len[i, :]-0.7))) * 1000) **2
    # min act

    for m in range(x.shape[0]):
        if m not in muscle_track_idx:
            j += weights["min_act"] * (x[m] ** 2)
        else:
            j += weights["activation_tracking"] * (((x[m]) - act[muscle_track_idx.index(m)] * scaling_factor[0]) ** 2)
    if not muscle_torque_as_constraint:
        mus_tau = _get_muscle_torque(x, q, qdot, p, p_mapping, muscle_casadi_function, scaling_factor, with_param)
    else:
        mus_tau = symbolics.muscle_torque
    # if muscle_torque_as_constraint:
    #     mus_tau_from_act = _get_muscle_torque(x, q, qdot, p, p_mapping, muscle_casadi_function, scaling_factor, with_param)
    #     #f = Function("msj", [x, q, qdot, p], [mus_tau_from_act])
    #     g =  mus_tau_from_act - symbolics.muscle_torque
    ignore_dof = [] if ignore_dof is None else ignore_dof
    pas_tau_tmp = pas_tau if with_torque else None
    count = 0
    for t in range(mus_tau.shape[0]):
        factor=1
        if with_torque and t in passive_torque_idx:
            to_substract = mus_tau[t] * scaling_factor[2] + pas_tau_tmp[count] if with_torque else mus_tau[t]
            count += 1
        elif t not in ignore_dof:
            to_substract = mus_tau[t] * scaling_factor[2]

        else:
            continue
        sqrt = 1 if tau_as_constraint else 2
        factor = 1 if t in [9] else 1
        factor = 1 if t in [15] else factor
        j += factor * weights["tau_tracking"] * ((tau[t] * scaling_factor[2] - to_substract)**2)
    return j, g

def return_obj_function(obj_funct, act, q, qdot, tau, emg, p=None, tau_res=None, tau_muscle=None, lm_init = None):
    obj_tmp = None
    if tau_res is not None:
        if tau_muscle is not None:
            obj_tmp = obj_funct(act, q,
                             qdot,
                             tau,
                             emg, tau_res, p,
                             tau_muscle, lm_init)
        else:
            obj_tmp = obj_funct(act, q, qdot, tau, emg, tau_res, p, lm_init)
    else:
        obj_tmp = obj_funct(act, q, qdot, tau, emg, p, lm_init)

    return obj_tmp

def get_cost_n_dependant(p, p_mapping, params_to_optim, scaling_factor,weights, use_ratio_tracking=False,  param_init=None,
                     bounds_l_norm=False, mx_variables=None, use_sx=False):
    lm_opti, lt_slack = None, None
    count = 0
    J_params = 0
    g = []
    for p_idx in range(len(p_mapping)):
        p_tmp = p[count: count + len(p_mapping[p_idx][0])]
        # for p_idx_bis in range(p_tmp.shape[0]):
        #     J_params += (weights[f"min_{params_to_optim[p_idx]}"] * (p_tmp[p_idx_bis] - 1 * scaling_factor[1][p_idx])) ** 2
        J_params += ca.sum1(weights[f"min_{params_to_optim[p_idx]}"] * ((
                    p_tmp - scaling_factor[1][p_idx]) ** 2))
        # J_params += ca.sum1(weights[f"min_{params_to_optim[p_idx]}"] * ((
        #             p_tmp ) ** 2))
        if params_to_optim[p_idx] == "lm_optim":
            lm_opti = p[count: count + len(p_mapping[p_idx][0])]
        elif params_to_optim[p_idx] == "lt_slack":
            lt_slack = p[count: count + len(p_mapping[p_idx][0])]
        count += len(p_mapping[p_idx][0])

    #norm_len = MX(muscle_len) / (lm_opti / MX(scaling_factor[1][params_to_optim.index("lm_optim")]))
    #for i in range(lm_opti.shape[0]):
    #    #J_params += (10000 * ca.sum2(norm_len[i, :] - 1)) **2
    #    J_params += ca.sum2((1-tanh(1000*(norm_len[i, :]-0.7))) * 1000) **2


        # for m in range(ratio_l.shape[0]):
        #     for k in range(ratio_l.shape[1]):
        #         if g is None:
        #             g = ratio_l[m, k]
        #         else:
        #             g += ratio_l[m, k]

    if use_ratio_tracking and "lt_slack" in params_to_optim and "lm_optim" in params_to_optim:
        for i in range(lm_opti.shape[0]):
            lm_init = param_init[params_to_optim.index("lm_optim")][i]
            lt_init = param_init[params_to_optim.index("lt_slack")][i]
            lm_optimized = lm_init * lm_opti[i]
            lt_slack_optimized = lt_init * lt_slack[i]
            ratio = lt_init / lm_init
            J_params += weights["ratio_tracking"] * ((lt_slack_optimized / lm_optimized - ratio) ** 2)
    #     J = _add_to_J(J, weights["ratio_tracking"], to_minimize)
    if isinstance(g, list) and len(g) == 0:
        g = None
    return J_params, g

def get_initial_values(model, passive_torque_idx, ns, muscle_track_idx, act, scaling_factor):
    tau_init = np.zeros((len(passive_torque_idx) * ns))
    x0 = np.zeros((model.nbMuscles() * ns, 1)) + 0.2 * scaling_factor[0]
    for i in range(ns):
        for m in range(model.nbMuscles()):
            if m in muscle_track_idx:
                idx = muscle_track_idx.index(m)
                x0[i * model.nbMuscles() + m] = act[idx, i] * scaling_factor[0]

    return x0, tau_init

def return_bounds(model, scaling_factor, p, ns, x, pas_tau, muscle_torque, x0, tau_0, with_param=True, with_torque=True,
                   p_init=None, params_to_optim=(), p_mapping=None, param_bounds=None, l_norm_bounded=False, g=None, tau_bounds=50):
    lbx = ca.DM.zeros(model.nbMuscles() * (ns)) + (0.0001) * scaling_factor[0]
    ubx = ca.DM.ones(model.nbMuscles() * (ns)) * scaling_factor[0]
    if g is not None:
        lbg = ca.DM.zeros(g.shape[0]) + 0.5
        ubg = ca.DM.zeros(g.shape[0]) + 1.5
    else:
        lbg = ca.DM.zeros(0)
        ubg = ca.DM.zeros(0)

    if l_norm_bounded:
        if "lm_optim" in params_to_optim:
            ubg = 1.5
            lbg = 0.5

    if with_param:
        lb_p = None
        ub_p = None
        init_p = None
        for p_idx, param in enumerate(params_to_optim):
            lb, ub = param_bounds[p_idx][0], param_bounds[p_idx][1]
            n_p = len(p_mapping[p_idx][0])
            init_p = np.array(p_init[p_idx]) * scaling_factor[1][p_idx] if init_p is None else ca.vertcat(init_p,
                                                                                             np.array(p_init[p_idx]) * scaling_factor[1][p_idx])
            lb_p = ca.DM.zeros(n_p) + lb * scaling_factor[1][p_idx] if lb_p is None else ca.vertcat(lb_p,
                                                                                             ca.DM.zeros(n_p) + lb *
                                                                                             scaling_factor[1][p_idx])
            ub_p = ca.DM.zeros(n_p) + ub * scaling_factor[1][p_idx] if ub_p is None else ca.vertcat(ub_p,
                                                                                             ca.DM.zeros(n_p) + ub *
                                                                                             scaling_factor[1][p_idx])
        lbx = ca.vertcat(lbx, lb_p)
        ubx = ca.vertcat(ubx, ub_p)
        x0 = ca.vertcat(x0, init_p)
        x = ca.vertcat(x, p)
    if with_torque:
        lb_tau = ca.DM.ones(tau_0.shape[0]) * (-tau_bounds) * scaling_factor[2]
        ub_tau = ca.DM.ones(tau_0.shape[0]) * tau_bounds * scaling_factor[2]
        init_tau = tau_0
        lbx = ca.vertcat(lbx, lb_tau)
        ubx = ca.vertcat(ubx, ub_tau)
        x0 = ca.vertcat(x0, init_tau)
        x = ca.vertcat(x, pas_tau)
    if muscle_torque is not None:
        lb_tau = ca.DM.ones(muscle_torque.shape[0]) * -100 * scaling_factor[2]
        ub_tau  = ca.DM.ones(muscle_torque.shape[0]) * 100 * scaling_factor[2]
        init = ca.DM.zeros(muscle_torque.shape[0])
        x0 = ca.vertcat(x0, init)
        x = ca.vertcat(x, muscle_torque)
        lbx = ca.vertcat(lbx, lb_tau)
        ubx = ca.vertcat(ubx, ub_tau)
    return {"lbx": lbx, "ubx": ubx, "lbg": lbg, "ubg": ubg, "x0": x0, "x": x}


def _get_muscle_torque(x, q, qdot, p, p_mapping, muscle_casadi_function, scaling_factor, with_param=True):
    # track tau
    if with_param:
        p_tmp = None
        count = 0
        for p_idx in range(len(p_mapping)):
            n_p = len(p_mapping[p_idx][0])
            p_tmp = p[count:count + n_p] / scaling_factor[1][p_idx] if p_tmp is None else vertcat(p_tmp, p[count:count + n_p] / scaling_factor[1][p_idx])
            count += n_p
        mus_tau = muscle_casadi_function(x / scaling_factor[0], q, qdot, p_tmp)
    else:
        mus_tau = muscle_casadi_function(x / scaling_factor[0], q, qdot)
    return mus_tau

def _get_muscle_length(q, p, p_mapping, casadi_function, scaling_factor, with_param=True):
    if with_param:
        p_tmp = None
        count = 0
        for p_idx in range(len(p_mapping)):
            n_p = len(p_mapping[p_idx][0])
            p_tmp = p[count:count + n_p] / scaling_factor[1][p_idx] if p_tmp is None else vertcat(p_tmp, p[count:count + n_p] / scaling_factor[1][p_idx])
            count += n_p
        mus_tau = casadi_function(q, p_tmp)
    else:
        mus_tau = casadi_function(q)
    return mus_tau

def return_param_from_mapping(p_mapping, p_init):
    final_param_list = []
    count = 0
    for mapping in p_mapping:
        p_tmp = p_init[count:count + len(mapping[0])]
        count += len(mapping[0])
        final_param_list.append([p_tmp[mapping[0].index(m)] for m in mapping[1]])
    return final_param_list


def _interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(markers_depth.shape[0]):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def _interpolate_data_2d(data, shape):
    new_data = np.zeros((data.shape[0], shape))
    x = np.linspace(0, 100, data.shape[1])
    f_mark = interp1d(x, data)
    x_new = np.linspace(0, 100, int(new_data.shape[1]))
    new_data = f_mark(x_new)
    return new_data


def _remove_outliers(data):
    new_data = np.zeros_like(data)
    std_outliers = np.std(data, axis=0)
    return new_data

def process_cycles(all_results, peaks, interpolation_size=120, remove_outliers=False, key__to_check = ["q", "tau", "emg"]):
    kay_for_size = [key for key in all_results.keys() if "q" in key and isinstance(all_results[key], np.ndarray)][0]
    data_size = all_results[kay_for_size].shape[1]
    dic_tmp = {}
    for key2 in all_results.keys():
        if key2 == "cycle" or key2 == "rt_matrix" or key2 == "marker_names":
            continue
        array_tmp = None
        if not isinstance(all_results[key2], np.ndarray):
            dic_tmp[key2] = []
            continue
        for k in range(len(peaks) - 1):
            if peaks[k + 1] > data_size:
                break
            interp_function = _interpolate_data_2d if len(all_results[key2].shape) == 2 else _interpolate_data
            if array_tmp is None:
                array_tmp = interp_function(all_results[key2][..., peaks[k]:peaks[k + 1]], interpolation_size)
                array_tmp = array_tmp[None, ...]
            else:
                data_interp = interp_function(all_results[key2][..., peaks[k]:peaks[k + 1]], interpolation_size)
                array_tmp = np.concatenate((array_tmp, data_interp[None, ...]), axis=0)
        dic_tmp[key2] = array_tmp
    key_to_check = ["q", "tau", "emg"]
    if remove_outliers:
        for key in key_to_check:
            if key in dic_tmp.keys():
                dic_tmp[key] = _remove_outliers(dic_tmp[key])
    all_results["cycles"] = dic_tmp
    return all_results


def apply_params(model, param_list, params_to_optim, model_param_init=None, with_casadi=True, ratio=None):
    for k in range(model.nbMuscles()):
        if "f_iso" in params_to_optim:
            f_init = model.muscle(k).characteristics().forceIsoMax() if not model_param_init else \
            model_param_init[params_to_optim.index("f_iso")][k]
            param_tmp = param_list[params_to_optim.index("f_iso")][k]
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setForceIsoMax(f_init * param_tmp)
        if "lm_optim" in params_to_optim:
            l_init = model.muscle(k).characteristics().optimalLength() if not model_param_init else \
            model_param_init[params_to_optim.index("lm_optim")][k]
            param_tmp = param_list[params_to_optim.index("lm_optim")][k]
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            opt_l = l_init * param_tmp
            model.muscle(k).characteristics().setOptimalLength(opt_l)
            if ratio and "lt_slack" not in params_to_optim:
                model.muscle(k).characteristics().setTendonSlackLength(opt_l * ratio[k])
        if "lt_slack" in params_to_optim:
            l_init = model.muscle(k).characteristics().tendonSlackLength() if not model_param_init else \
            model_param_init[params_to_optim.index("lt_slack")][k]
            param_tmp = param_list[params_to_optim.index("lt_slack")][k]
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setTendonSlackLength(l_init * param_tmp)
    return model



def optimize_parameters_init(all_length, param_value):
    from scipy.optimize import minimize
    # Define the function with a penalty if the parameter goes out of the range
    def objective(x, lengths, lower_bound, upper_bound):
        # If x is within bounds, return zero penalty (objective is zero)
        norm_len = (lengths / (param_value * x))
        if lower_bound <= norm_len.min() <= upper_bound and lower_bound <= norm_len.max() <= upper_bound:
            return 0
        # Apply a penalty proportional to the distance from the nearest bound if out of range
        elif lower_bound >= norm_len.min():
           return  (lower_bound - norm_len.min()) ** 2
        elif upper_bound <= norm_len.max():
            return (norm_len.max() - upper_bound) ** 2
        else:
            return 0


    # Set bounds and initial guess
    lower_bound = 0.5
    upper_bound = 1.5
    initial_guess = 1.0  # Initial guess outside the range

    # Run optimization to find the parameter value within bounds
    result = minimize(objective, initial_guess, args=(all_length, lower_bound, upper_bound), bounds=[(lower_bound, upper_bound)])

    # Extract optimized parameter
    optimized_parameter = result.x[0]
    return optimized_parameter

