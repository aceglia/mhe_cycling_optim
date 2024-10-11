from biosiglive import load
import numpy as np


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
                            p_init=None
                            ) -> MX:
    if with_param:
        param_list = _return_param_from_mapping(p_mapping, p_init)
        model = _apply_params(model, param_list, params_to_optim, model_params_init, ratio=ratio)
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(activations[k])
    # muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx() * p_init[:model.nbMuscles()]
    # mjt = model.muscularJointTorque(muscles_force, q, qdot).to_mx()
    mjt = model.muscularJointTorque(muscles_states, q, qdot).to_mx()
    return mjt


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


