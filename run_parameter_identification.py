from optim_params.identification_utils import prepare_data
from optim_params.parameters_identifier import ParametersIdentifier
from optim_params.identification_utils import process_cycles
from optim_params.enum import Parameters
import numpy as np

def _prepare_data(ocp_result, suffix, cycle, em_delay, peaks, n_frame_cycle, rate=120, ratio=1, random_idx_list=None):
    em_delay_frame = int(em_delay * rate)
    if em_delay_frame != 0:
        for key in ocp_result.keys():
            if "q" in key or "qdot" in key or "tau" in key or "f_ext" in key:
                ocp_result[key] = ocp_result[key][:, em_delay_frame:]
            if "emg" in key:
                ocp_result[key] = ocp_result[key][:, :-em_delay_frame] if em_delay != 0 else ocp_result[key][..., :]
    ocp_result = process_cycles(ocp_result, peaks, interpolation_size=rate, remove_outliers=False)
    q, qdot, tau, f_ext, emg_proc = ocp_result["cycles"]["q" + suffix], ocp_result["cycles"]["qdot" + suffix], ocp_result["cycles"]["tau" + suffix], ocp_result["cycles"]["f_ext"], ocp_result["cycles"]["emg"]
    if cycle > q.shape[0] - 1:
        raise ValueError("cycle should be less than the number of cycles")
    q, qdot, tau, f_ext, emg_proc = q[random_idx_list, ...], qdot[random_idx_list, ...], tau[random_idx_list, ...], f_ext[random_idx_list, ...], emg_proc[random_idx_list, ...]
    q_final = np.zeros((q.shape[1], n_frame_cycle * cycle))
    qdot_final = np.zeros((qdot.shape[1], n_frame_cycle  * cycle))
    tau_final = np.zeros((tau.shape[1], n_frame_cycle * cycle))
    f_ext_final = np.zeros((f_ext.shape[1], n_frame_cycle  * cycle))
    emg_proc_final = np.zeros((emg_proc.shape[1], n_frame_cycle  * cycle))
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

weights = {"tau_tracking": 2,
           "activation_tracking": 10,
           "min_act": 1,
           "min_f_iso": 5,
           "min_lm_optim": 5,
           "min_lt_slack": 100,
           "min_pas_torque": 0.6,
           "ratio_tracking": 1,
           "dynamics": 100}

if __name__ == '__main__':
    params_to_optimize = [Parameters.f_iso, Parameters.l_optim]
    identifier = ParametersIdentifier(params_to_optimize)
    identifier.load_experimental_data(prepare_data_function=prepare_data)