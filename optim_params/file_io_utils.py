from biosiglive import load, save
import os

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
    for part in participants:
        all_files = os.listdir(f"{data_dir}{os.sep}{part}")
        all_files = [file for file in all_files if any([ext in file for ext in to_include]) and not any([ext in file for ext in to_exclude])]
        all_files = [f"{data_dir}{os.sep}{part}{os.sep}{file}" for file in all_files]
        final_files = all_files if not trial_names else []
        if trial_names:
            for trial in trial_names[participants.index(part)]:
                for file in all_files:
                    if trial in file:
                        final_files.append(file)
                        break
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

