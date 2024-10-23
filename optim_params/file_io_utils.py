from biosiglive import load, save
import os
from optim_params.identification_utils import process_cycles, generate_random_idx
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def get_data_dict(file_path, n_cycles=1, batch_size=1, rate=120, cycle_size=60, from_id=False,  em_delay=0):
    suffix = "_ocp" if not from_id else "_id"
    ocp_result = load(file_path)
    peaks = find_peaks(ocp_result["q" + suffix][-2, :], height=0.5)[0]
    em_delay_frame = int(em_delay * rate)
    if em_delay_frame != 0:
        for key in ocp_result.keys():
            if "q" in key or "q_dot" in key or "tau" in key or "f_ext" in key:
                ocp_result[key] = ocp_result[key][:, em_delay_frame:]
            if "emg" in key:
                ocp_result[key] = ocp_result[key][:, :-em_delay_frame] if em_delay != 0 else ocp_result[key][..., :]
    ocp_result = process_cycles(ocp_result, peaks, interpolation_size=cycle_size, remove_outliers=False)
    q, qdot, tau, f_ext, emg = (ocp_result["cycles"]["q" + suffix], ocp_result["cycles"]["q_dot" + suffix],
                                     ocp_result["cycles"]["tau" + suffix], ocp_result["cycles"]["f_ext"],
                                     ocp_result["cycles"]["emg"])
    if n_cycles > q.shape[0] - 1:
        raise ValueError("The number of selected cycles should be less than the number of total cycles")
    random_idx_list = generate_random_idx(n_cycles, batch_size, q.shape[0])
    dict_to_return = {"q": q, "qdot": qdot, "tau": tau, "f_ext": f_ext, "emg": emg}
    return dict_to_return, random_idx_list


def get_experimental_data(file_path, n_start=0, n_stop=None, source="dlc_1", downsample=2):
    """
    This function loads the experimental data from a file and returns it as a dictionary.
    :param file_path:
    :return:
    """
    out_dict = {}
    data = load(file_path)
    if n_stop is None:
        n_stop = data["emg"].shape[1]
    out_dict["f_ext"] = data["f_ext"][:, n_start:n_stop][..., ::downsample]
    out_dict["q_init"] = data[source]["q"][:, n_start:n_stop][..., ::downsample]
    out_dict["q_dot_init"] = data[source]["q_dot"][:, n_start:n_stop][..., ::downsample]
    out_dict["emg"] = data["emg"][:, n_start:n_stop][..., ::downsample]
    names = data[source]["marker_names"]
    ia_idx = names.index("SCAP_IA")
    ts_idx = names.index("SCAP_TS")
    mark_ia = data[source][f"tracked_markers"][:, ia_idx, :].copy()
    mark_ts = data[source][f"tracked_markers"][:, ts_idx, :].copy()
    data[source][f"tracked_markers"][:, ia_idx, :] = mark_ts
    data[source][f"tracked_markers"][:, ts_idx, :] = mark_ia
    out_dict["markers_target"] = data[source][f"tracked_markers"][:, :, n_start:n_stop][..., ::downsample]
    return out_dict

def get_all_file(participants, data_dir, trial_names=None, to_include=(), to_exclude=()):
    all_path = []
    parts = []
    if trial_names and len(trial_names) != len(participants):
        trial_names = [trial_names for _ in participants]
    for p, part in enumerate(participants):
        try:
            all_files = os.listdir(f"{data_dir}{os.sep}{part}")
        except FileNotFoundError:
            print(f"Participant {part} not found in {data_dir}")
            continue
        if trial_names:
            to_include += trial_names[p] if isinstance(trial_names[p], list) else trial_names
        all_files = [file for file in all_files if all([ext in file for ext in to_include]) and not any([ext in file for ext in to_exclude])]
        final_files = [f"{data_dir}{os.sep}{part}{os.sep}{file}" for file in all_files]
        parts.append([part for _ in final_files])
        all_path.append(final_files)
    return sum(all_path, []), sum(parts, [])

def save_torque_estimation(file_path, torque_estimation):
    """
    This function saves the torque estimation to a file.
    :param file_path:
    :param torque_estimation:
    :return:
    """
    empty_dict = {"q_ocp": merged_states["q"],
                "qdot_ocp": merged_states["qdot"],
                "tau_ocp": merged_controls["tau"],
                "q_init": kalman_data[source]["q_raw"][:, n_start:n_stop][..., ::2][:, : - (n_shooting + 1)],
                "qdot_init": kalman_data[source]["q_dot"][:, n_start:n_stop][..., ::2][:, : - (n_shooting + 1)],
                # "tau_init": kalman_data[source]["tau"][:, n_start:n_stop - (n_shooting + 1)],
                "q_id": q_id[:, :- (n_shooting)],
                "qdot_id": q_dot_id[:, :- (n_shooting)],
                "qddot_id": q_ddot_id[:, :- (n_shooting)],
                "tau_id": tau_id[:, :- (n_shooting)],
                "markers": markers_init[:, :, :-(n_shooting + 1)],
                "emg": kalman_data["emg"][:, n_start:n_stop][..., ::2][:, :- (n_shooting + 1)],
                "peaks": peaks,
                "use_mhe": with_mhe,
                "n_shooting": n_shooting,
                "ocp_time": final_time,
                "n_start": n_start,
                "n_stop": n_stop,
                "f_ext": None,
                "total_time_mhe": time.time() - tic,
                }

    if with_f_ext:
        save_dic["f_ext_ocp"] = merged_controls["f_ext"]
        save_dic["init_f_ext"] = f_ext[:, :-(n_shooting + 1)]

