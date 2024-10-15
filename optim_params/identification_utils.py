from biosiglive import load
import numpy as np
from casadi import Function, vertcat


def get_initial_values(model, params_to_optim):
    model_param_init = []
    for p in params_to_optim:
        if p == "f_iso":
            model_param_init.append(
                [model.muscle(k).characteristics().forceIsoMax().to_mx() for k in range(model.nbMuscles())])
        elif p == "lm_optim":
            model_param_init.append(
                [model.muscle(k).characteristics().optimalLength().to_mx() for k in range(model.nbMuscles())])
        elif p == "lt_slack":
            model_param_init.append(
                [model.muscle(k).characteristics().tendonSlackLength().to_mx() for k in range(model.nbMuscles())])
    ratio = [model.muscle(k).characteristics().tendonSlackLength().to_mx() / model.muscle(
        k).characteristics().optimalLength().to_mx() for k in range(model.nbMuscles())]
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


def get_cost_n_dependant(scaling_factor, symbolics, weights,
                           p_mapping=None,
                           with_torque=True,
                           muscle_track_idx=None,
                           muscle_casadi_function=None,
                           with_param=True,
                           passive_torque_idx=None,
                           tau_as_constraint=False,
                          ignore_dof = None
                           ):
    j = 0
    x, q, qdot = symbolics.x, symbolics.q, symbolics.qdot
    p, act, tau, pas_tau = symbolics.p, symbolics.emg, symbolics.tau, symbolics.pas_tau
    if with_torque:
        torque_weights = np.array([weights["min_pas_torque"] for _ in range(pas_tau.shape[0])])
        j = [j + (torque_weights[tau_idx] * pas_tau[tau_idx]) ** 2 for tau_idx in range(pas_tau.shape[0])]

    # min act
    j = [j + (weights["min_act"] * x[m]) ** 2 for m in range(x.shape[0]) if m not in muscle_track_idx]
    j = [j + (weights["activation_tracking"] * ((x[m]) - act[m])) ** 2 for m in range(x.shape[0]) if m in muscle_track_idx]

    mus_tau = _get_muscle_torque(x, q, qdot, p, p_mapping, muscle_casadi_function, scaling_factor, with_param)
    pas_tau_tmp = pas_tau if with_torque else None
    count = 0
    for t in range(mus_tau.shape[0]):
        if t in passive_torque_idx:
            to_substract = mus_tau[t] * scaling_factor[2] + pas_tau_tmp[count] if with_torque else mus_tau[t]
            count += 1
        elif t not in ignore_dof:
            to_substract = mus_tau[t] * scaling_factor[2]
        else:
            continue
        sqrt = 1 if tau_as_constraint else 2
        factor = 0.3 if t == 3 else 1
        j += (weights["tau_tracking"] * factor * (tau[t] * scaling_factor[2] - to_substract)) ** sqrt
    return j


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

def return_param_from_mapping(p_mapping, p_init):
    final_param_list = []
    count = 0
    for mapping in p_mapping:
        p_tmp = 0
        for m in mapping[1]:
            p_tmp = vertcat(p_tmp, p_init[mapping[0].index(m) + count])
        final_param_list.append(p_tmp)
        count += len(mapping[0])
    return final_param_list


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
            model.muscle(k).characteristics().setOptimalLength(l_init * param_tmp)
            if ratio and "lt_slack" not in params_to_optim:
                opt_l = model.muscle(k).characteristics().optimalLength().to_mx() if with_casadi else float(
                    model.muscle(k).characteristics().optimalLength())
                model.muscle(k).characteristics().setTendonSlackLength(opt_l * ratio[k])
        if "lt_slack" in params_to_optim:
            l_init = model.muscle(k).characteristics().tendonSlackLength() if not model_param_init else \
            model_param_init[params_to_optim.index("lt_slack")][k]
            param_tmp = param_list[params_to_optim.index("lt_slack")][k]
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setTendonSlackLength(l_init * param_tmp)
    return model


def prepare_data(q, q_dot, tau, emg, em_delay=None, data_rate=100, peaks=None):
    em_delay_frame = int(em_delay * data_rate)
    if em_delay_frame != 0:
        for key in ocp_result.keys():
            if "q" in key or "qdot" in key or "tau" in key or "f_ext" in key:
                ocp_result[key] = ocp_result[key][:, em_delay_frame:]
            if "emg" in key:
                ocp_result[key] = ocp_result[key][:, :-em_delay_frame] if em_delay != 0 else ocp_result[key][..., :]
    ocp_result = process_cycles(ocp_result, peaks, interpolation_size=rate, remove_outliers=False)
    q, qdot, tau, f_ext, emg_proc = ocp_result["cycles"]["q" + suffix], ocp_result["cycles"]["qdot" + suffix], \
        ocp_result["cycles"]["tau" + suffix], ocp_result["cycles"]["f_ext"], ocp_result["cycles"]["emg"]
    if cycle > q.shape[0] - 1:
        raise ValueError("cycle should be less than the number of cycles")
    q, qdot, tau, f_ext, emg_proc = q[random_idx_list, ...], qdot[random_idx_list, ...], tau[random_idx_list, ...], \
        f_ext[random_idx_list, ...], emg_proc[random_idx_list, ...]
    q_final = np.zeros((q.shape[1], n_frame_cycle * cycle))
    qdot_final = np.zeros((qdot.shape[1], n_frame_cycle * cycle))
    tau_final = np.zeros((tau.shape[1], n_frame_cycle * cycle))
    f_ext_final = np.zeros((f_ext.shape[1], n_frame_cycle * cycle))
    emg_proc_final = np.zeros((emg_proc.shape[1], n_frame_cycle * cycle))
    for i in range(cycle):
        #     q_filtered = OfflineProcessing().butter_lowpass_filter(q[i, :],
        #                                                            6, 60, 2)
        #     qdot_filtered = OfflineProcessing().butter_lowpass_filter(qdot[i, :],
        #                                                            6, 60, 2)
        #     tau_filtered = OfflineProcessing().butter_lowpass_filter(tau[i, :],
        #                                                            6, 60, 2)
        q_final[:, i * n_frame_cycle:(i + 1) * n_frame_cycle] = q[i, :, ::ratio]
        qdot_final[:, i * n_frame_cycle:(i + 1) * n_frame_cycle] = qdot[i, :, ::ratio]
        tau_final[:, i * n_frame_cycle:(i + 1) * n_frame_cycle] = tau[i, :, ::ratio]
        f_ext_final[:, i * n_frame_cycle:(i + 1) * n_frame_cycle] = f_ext[i, :, ::ratio]
        emg_proc_final[:, i * n_frame_cycle:(i + 1) * n_frame_cycle] = emg_proc[i, :, ::ratio]
    return q_final, qdot_final, tau_final, f_ext_final, emg_proc_final, random_idx_list


