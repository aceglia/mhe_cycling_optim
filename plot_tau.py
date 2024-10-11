import matplotlib.pyplot as plt
import numpy as np
from biosiglive import load
import biorbd
from biosiglive import OfflineProcessing, MskFunctions, InverseKinematicsMethods
from mhe.utils import apply_params
from scipy.signal import find_peaks
import os
from scipy.interpolate import interp1d
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

def plot_results(x_ref, states_tmp, controls_tmp, muscles_target, f_ext_target, muscle_track_idx, nbQ):
    states = {}
    controls = {}
    for key in states_tmp[0].keys():
        states[key] = np.concatenate([states_tmp[i][key] for i in range(len(states_tmp))], axis=1)
    for key in controls_tmp[0].keys():
        controls[key] = np.concatenate([controls_tmp[i][key] for i in range(len(controls_tmp))], axis=1)

    for key in states:
        plt.figure(f"states_{key}")
        for i in range(states[key].shape[0]):
            plt.subplot(4, int(states[key].shape[0] // 4) + 1, i + 1)
            if key == "q":
                plt.plot(x_ref[i, :states[key].shape[1]], "r")
            if key == "qdot":
                plt.plot(x_ref[i+nbQ, :states[key].shape[1]], "r")
            plt.plot(states[key][i, :])

    muscle_idx = muscle_track_idx
    for key in controls:
        plt.figure(f"controls_{key}")
        for i in range(controls[key].shape[0]):
            plt.subplot(4, int(controls[key].shape[0] // 4) + 1, i + 1)
            if key == "muscles":
                if i in muscle_idx:
                    plt.plot(muscles_target[muscle_idx.index(i), :controls[key].shape[1]], "r")
            if key == "f_ext":
                plt.plot(f_ext_target[i, :controls[key].shape[1]], "r")
            plt.plot(controls[key][i, :])
        plt.show()


def get_muscular_torque(x, act, model, parameters_file_path=None, with_casadi=False, ratio = True):
    """
    Get the muscular torque.
    """
    if parameters_file_path:
        model = apply_params(model, parameters_file_path, with_casadi=with_casadi, ratio=ratio)
    muscular_torque = np.zeros((model.nbQ(), x.shape[1]))
    states = model.stateSet()  # Get the muscle state set
    for i in range(act.shape[1]):
        for a, state in zip(act[:, i], states):
            state.setActivation(a)  # And fill it with the current value
        muscular_torque[:, i] = model.muscularJointTorque(
            states, x[: model.nbQ(), i], x[model.nbQ() : model.nbQ() * 2, i]
        ).to_array()
    return muscular_torque


def get_id_torque(q, q_dot, model=None, f_ext=None, rate=60):
    # q_init = x[: model.nbQ(), :]
    # qdot = x[model.nbQ(): model.nbQ() * 2, :]
    # qddot = x[model.nbQ() * 2: model.nbQ() * 3, :]
    # q_filtered = OfflineProcessing().butter_lowpass_filter(q_init,
    #                                                        6, rate, 2)
    # qdot_new = np.zeros_like(q_init)
    # qdot_new[:, 1:-1] = (q_filtered[:, 2:] - q_filtered[:, :-2]) / (2 / rate)
    # qdot_new[:, 0] = q_filtered[:, 1] - q_filtered[:, 0]
    # qdot_new[:, -1] = q_filtered[:, -1] - q_filtered[:, -2]
    q_filtered = q
    qdot_new = q_dot
    # for i in range(1, q_filtered.shape[1] - 2):
    #     qdot_new[:, i] = (q_filtered[:, i + 1] - q_filtered[:, i - 1]) / (2 / 120)
    qddot_new = np.zeros_like(qdot_new)
    qddot_new[:, 1:-1] = (qdot_new[:, 2:] - qdot_new[:, :-2]) / (2 / rate)
    qddot_new[:, 0] = qdot_new[:, 1] - qdot_new[:, 0]
    qddot_new[:, -1] = qdot_new[:, -1] - qdot_new[:, -2]
    q, qdot, qddot = q_filtered, qdot_new, qddot_new
    # qddot = OfflineProcessing(data_rate=120, processing_window=q.shape[1]).butter_lowpass_filter(qddot, 2, 120, 4)

    tau_from_b = np.zeros((model.nbQ(), q.shape[1]))
    for i in range(q.shape[1]):
        B = [0, 0, 0, 1]
        f_ext_mat = np.zeros((6, 1))
        all_jcs = model.allGlobalJCS(q[:, i])
        RT = all_jcs[-1].to_array()
        B = RT @ B
        vecteur_OB = B[:3]
        f_ext_mat[:3, 0] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])
        f_ext_mat[3:, 0] = f_ext[3:, i]
        ext_load = model.externalForceSet()
        ext_load.add("hand_left", f_ext_mat[:, 0])
        tau_from_b[:, i] = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i], ext_load).to_array()
    return tau_from_b

def get_tracking_idx(model, emg_names):
    muscle_list = []
    for i in range(model.nbMuscles()):
        muscle_list.append(model.muscleNames()[i].to_string())
    muscle_track_idx = []
    for i in range(len(emg_names)):
        for j in range(len(muscle_list)):
            if emg_names[i] in muscle_list[j]:
                muscle_track_idx.append(j)
    return muscle_track_idx

def plot_all_window(data_path, n_windows=None, plot_by_windows=False, line_style="-", color="b", model_path=None):
    data = load(data_path, merge=False)
    key_to_plot = ["q_est", "dq_est", "u_est", "muscle_force", "tau_est", "f_ext"]
    ref_key_to_plot = ["q_ref", "dq_ref", "muscles_target", None, None, "f_ext_ref"]
    initial_guess_key = ["q_init", "dq_init", "u_init", None, "tau_init", "f_ext_init"]
    kin_target = None
    target = []
    n_mhe = data[0]["n_mhe"]
    n_q = data[0]["x_init"]["q"].shape[0]
    n_frame_to_plot = 0
    initial_guess = [{}] * len(key_to_plot) if not plot_by_windows else []
    target = [{}] * len(key_to_plot) if not plot_by_windows else []
    data_to_plot = {}
    for i in range(len(data)):
        if not plot_by_windows:
            for key in key_to_plot:
                if data[i][key] is None:
                    data_to_plot[key] = None
                    continue

                data_to_plot[key] = data[i][key][..., n_frame_to_plot:n_frame_to_plot+1] if i == 0 else np.append(data_to_plot[key], data[i][key][..., n_frame_to_plot:n_frame_to_plot+1] ,axis=-1)
            if i == 0:
                initial_guess[0]["q_init"] = np.array(data[i]["x_init"]["q"])[..., n_frame_to_plot:n_frame_to_plot+1]
                initial_guess[1]["dq_init"] = np.array(data[i]["x_init"]["qdot"])[..., n_frame_to_plot:n_frame_to_plot+1]
                initial_guess[2]["u_init"] = np.array(data[i]["u_init"]["muscles"])[..., n_frame_to_plot:n_frame_to_plot+1]
                initial_guess[4]["tau_init"] = np.array(data[i]["u_init"]["tau"])[..., n_frame_to_plot:n_frame_to_plot+1]
                initial_guess[5]["f_ext_init"] = None if "f_ext" not in data[i]["u_init"].keys() else np.array(data[i]["u_init"]["f_ext"])[..., n_frame_to_plot:n_frame_to_plot+1]
                target[0]["q_ref"] = data[i]["x_ref"][:n_q, n_frame_to_plot:n_frame_to_plot+1]
                target[1]["dq_ref"] = data[i]["x_ref"][n_q:, n_frame_to_plot:n_frame_to_plot+1]
                target[2]["muscles_target"] = data[i]["muscles_target"][:, n_frame_to_plot:n_frame_to_plot+1]
                target[4]["tau_est"] = None
                target[5]["f_ext_ref"] = data[i]["f_ext_ref"][:, n_frame_to_plot:n_frame_to_plot+1]
            else:
                initial_guess[0]["q_init"] = np.append(initial_guess[0]["q_init"], np.array(data[i]["x_init"]["q"])[:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                initial_guess[1]["dq_init"] = np.append(initial_guess[1]["dq_init"], np.array(data[i]["x_init"]["qdot"])[:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                initial_guess[2]["u_init"] = np.append(initial_guess[2]["u_init"], np.array(data[i]["u_init"]["muscles"])[:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                initial_guess[4]["tau_init"] = np.append(initial_guess[4]["tau_init"], np.array(data[i]["u_init"]["tau"])[:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                initial_guess[5]["f_ext_init"] = None if "f_ext" not in data[i]["u_init"].keys() else np.append(initial_guess[5]["f_ext_init"], np.array(data[i]["u_init"]["f_ext"])[:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                target[0]["q_ref"] = np.append(target[0]["q_ref"], data[i]["x_ref"][:n_q, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                target[1]["dq_ref"] = np.append(target[1]["dq_ref"], data[i]["x_ref"][n_q:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                target[2]["muscles_target"] = np.append(target[2]["muscles_target"], data[i]["muscles_target"][:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
                target[5]["f_ext_ref"] = np.append(target[5]["f_ext_ref"], data[i]["f_ext_ref"][:, n_frame_to_plot:n_frame_to_plot+1], axis=-1)
        else:
            initial_guess.append({})
            initial_guess[-1]["q_init"] = np.array(data[i]["x_init"]["q"])
            initial_guess[-1]["dq_init"] = np.array(data[i]["x_init"]["qdot"])
            initial_guess[-1]["u_init"] = np.array(data[i]["u_init"]["muscles"])
            initial_guess[-1]["tau_init"] = np.array(data[i]["u_init"]["tau"])
            initial_guess[-1]["f_ext_init"] = None if "f_ext" not in data[i]["u_init"].keys() else np.array(data[i]["u_init"]["f_ext"])
            target.append({})
            target[-1]["q_ref"] = data[i]["x_ref"][:n_q, :]
            target[-1]["dq_ref"] = data[i]["x_ref"][n_q:, :]
            target[-1]["muscles_target"] = data[i]["muscles_target"]
            target[-1]["f_ext_ref"] = data[i]["f_ext_ref"]
    data_to_plot = [data_to_plot]
    slide_size = data[0]["slide_size"]
    n_iter = len(data) if n_windows is None else n_windows
    for i in range(len(data)):
        key = "kin_target"
        kin_target = data[i][key][..., n_frame_to_plot:n_frame_to_plot + 1] if i == 0 else np.append(
            kin_target, data[i][key][..., n_frame_to_plot:n_frame_to_plot + 1], axis=-1)

    msk = MskFunctions(model_path, data_buffer_size=kin_target.shape[2], system_rate=120)
    model = msk.model
    markers = np.ndarray((3, model.nbMarkers(), data_to_plot[0]["q_est"].shape[1]))
    for i in range(data_to_plot[0]["q_est"].shape[1]):
        markers[:, :, i] = np.array(
            [mark.to_array() for mark in model.markers(data_to_plot[0]["q_est"][:, i])]
        ).T
    q, qdot, qddot = msk.compute_inverse_kinematics(kin_target, InverseKinematicsMethods.BiorbdKalman,
                                                    kalman_freq=120)
    # q = data_to_plot[0]["q_est"]
    # qdot = data_to_plot[0]["dq_est"]
    # qddot = np.zeros_like(q)
    # for i in range(1, q.shape[1] - 1):
    #     qddot[:, i] = (qdot[:, i+1] - qdot[:, i-1]) / (1/120)

    # import bioviz
    # b = bioviz.Viz(loaded_model=model)
    # b.load_movement(data_to_plot[0]["q_est"])
    # b.load_experimental_markers(kin_target)
    # b.exec()
    q_est = np.array([k["q_est"][:, 0] for k in data]).T
    tau = get_id_torque(np.concatenate((q_est, qdot, qddot), axis=0), msk.model, data_to_plot[0]["f_ext"])
    mus_tau = get_muscular_torque(np.concatenate((data_to_plot[0]["q_est"], data_to_plot[0]["dq_est"]), axis=0),
                                  data_to_plot[0]["u_est"], msk.model)
    plt.figure("markers")
    for m in range(markers.shape[1]):
        plt.subplot(4, markers.shape[1] // 4 + 1, m+1)
        for i in range(3):
            plt.plot(kin_target[i, m, :], "r", label="ref", alpha=0.5)
            plt.plot(markers[i, m, :], line_style, c=color, label="optim")
    plt.figure("tau_tot")


    for m in range(tau.shape[0]):
        plt.subplot(4, tau.shape[0] // 4 + 1, m+1)
        plt.plot(tau[m, :], "r", label="id")
        plt.plot(mus_tau[m, :] + data_to_plot[0]["tau_est"][m, :],line_style, c=color, label="optim")
    #
    tau_tot = np.array([max(tau) for tau in np.abs(mus_tau+ data_to_plot[0]["tau_est"])])
    tau_est = data_to_plot[0]["tau_est"] / np.repeat(tau_tot[:, None], data_to_plot[0]["tau_est"].shape[1], axis=1)
    plt.figure("norm_tau")
    for m in range(tau.shape[0]):
        plt.subplot(4, tau.shape[0] // 4 + 1, m+1)
        plt.plot(tau_est[m, :], line_style, c=color)

    # import bioviz
    # b = bioviz.Viz(loaded_model=model)
    # b.load_movement(data_to_plot[0]["q_est"])
    # b.load_experimental_markers(markers)
    # b.exec()

    emg_names = ["PectoralisMajorThorax",
                 "BIC",
                 "TRI",
                 "LatissimusDorsi",
                 'TrapeziusScapula_S',
                 #'TrapeziusClavicle',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                  'DeltoideusScapula_P']
    track_idx = get_tracking_idx(model, emg_names)

    n_iter = 1 if not plot_by_windows else n_iter
    for k, key in enumerate(key_to_plot):
        if data[0][key] is None:
            continue
        if isinstance(data[0][key], list):
            continue
        plt.figure(f"{key}")
        for n in range(n_iter):
            data_tmp = data[n][key] if plot_by_windows else data_to_plot[n][key]
            x_tmp = np.arange(n * slide_size, n * slide_size + data_tmp.shape[-1], 1) if plot_by_windows else np.arange(data_tmp.shape[-1])
            # make a x axis fill of int from 0 to n_iter * slide_size
            for j in range(data_tmp.shape[0]):
                plt.subplot(4, int(data_tmp.shape[0] // 4) + 1, j + 1)
                plt.plot(x_tmp, data_tmp[j, :], line_style, c=color)
                if ref_key_to_plot[k] is not None:
                    if plot_by_windows:
                        if target[n][ref_key_to_plot[k]].shape[-1] != data_tmp.shape[-1]:
                            target[n][ref_key_to_plot[k]] = np.repeat(target[n][ref_key_to_plot[k]], data_tmp.shape[-1], axis=-1)
                        plt.plot(x_tmp, target[n][ref_key_to_plot[k]][j, :], "r")
                    else:
                        if ref_key_to_plot[k] == "muscles_target":
                            if j in track_idx:
                                plt.plot(x_tmp, target[k][ref_key_to_plot[k]][track_idx.index(j), :], "r")
                        else:
                            plt.plot(x_tmp, target[k][ref_key_to_plot[k]][j, :], "r")
                if key in ["u_est", "muscle_force"]:
                    plt.title(model.muscleNames()[j].to_string())
                if key in ["q_est", "dq_est","tau_est"]:
                    plt.title(model.nameDof()[j].to_string())
                # if initial_guess_key[k] is not None:
                #     if plot_by_windows:
                #         if initial_guess[n][initial_guess_key[k]].shape[-1] != data_tmp.shape[-1]:
                #             initial_guess[n][initial_guess_key[k]] = np.repeat(initial_guess[n][initial_guess_key[k]], data_tmp.shape[-1], axis=-1)
                #             init_guess = initial_guess[n][initial_guess_key[k]]
                #         else:
                #             init_guess = initial_guess[n][initial_guess_key[k]]
                #     else:
                #         init_guess = initial_guess[k][initial_guess_key[k]]
                    #plt.plot(x_tmp, init_guess[j, :], "g")
    # plt.show()

def process_cycles(all_results, peaks, n_peaks=None, key_for_size="q_est"):
    dic_tmp = {}
    data_size = all_results[key_for_size].shape[1]
    all_results["cycles"] = {}
    for key2 in all_results.keys():
        if key2 == "cycles":
            continue
        array_tmp = None
        if n_peaks and n_peaks > len(peaks) - 1:
            raise ValueError("n_peaks should be less than the number of peaks")
        for k in range(len(peaks) - 1):
            if peaks[k + 1] > data_size:
                break
            interp_function = _interpolate_data_2d if len(all_results[key2].shape) == 2 else _interpolate_data
            if array_tmp is None:
                array_tmp = interp_function(all_results[key2][..., peaks[k]:peaks[k + 1]], 100)
                array_tmp = array_tmp[None, ...]
            else:
                data_interp = interp_function(all_results[key2][..., peaks[k]:peaks[k + 1]], 100)
                array_tmp = np.concatenate((array_tmp, data_interp[None, ...]), axis=0)
        all_results["cycles"][key2] = array_tmp
    return all_results

def plot_cycles(model, data_path, idx_to_export=0, cycles=True,color=None, line_style=None, optim_param_path=None, mhe_file=None, compare_to_fd=False):
    data = load(data_path, merge=False)
    data_mhe = None
    if compare_to_fd:
        if mhe_file:
            key_to_keep = ["q", "qdot", "tau"]
            data_mhe = load(mhe_file)
            data_mhe_tmp = {}
            for key in key_to_keep:
                data_mhe_tmp[key] = data_mhe[key]
            peaks = find_peaks(data_mhe["q"][-2, :])[0]
            data_mhe = process_cycles(data_mhe_tmp, peaks, key_for_size="q")
        else:
            raise RuntimeError("No mhe file provided")
    key_to_export = ["q_est", "dq_est", "u_est", "tau_est", "muscles_target", "f_ext", "muscle_force"]
    dic_merged = {}
    for key in data[0].keys():
        if key in key_to_export:
            dic_merged[key] = np.array([k[key][..., idx_to_export] for k in data]).T
    bio_model = biorbd.Model(model)
    q_est = dic_merged["q_est"]
    dq_est = dic_merged["dq_est"]
    u_est = dic_merged["u_est"]
    dic_merged["mus_tau"] = get_muscular_torque(np.concatenate((q_est, dq_est), axis=0),
                                  u_est, bio_model, parameters_file_path=optim_param_path) + dic_merged["tau_est"]
    dic_merged["tau"] = get_id_torque(q_est, dq_est, bio_model, dic_merged["f_ext"], rate=60)

    key_to_export.append("mus_tau")
    key_to_export.append("tau")

    if cycles:
        peaks = find_peaks(q_est[-2, :])[0]
        # plt.figure("peaks")
        # plt.plot(q_est[-2, :])
        # plt.scatter(peaks, q_est[-2, peaks])
        # plt.show()
        dic_merged = process_cycles(dic_merged, peaks)
    key_to_export.pop(key_to_export.index("muscles_target"))
    key_to_export.pop(key_to_export.index("tau"))
    if compare_to_fd:
        dic_merged["tau"] = data_mhe["tau"]
        dic_merged["cycles"]["tau"] = data_mhe["cycles"]["tau"]
        dic_merged["cycles"]["q_dot_ref"] = data_mhe["cycles"]["qdot"]
        dic_merged["q_dot_ref"] = data_mhe["qdot"]
        dic_merged["cycles"]["q_ref"] = data_mhe["cycles"]["q"]
        dic_merged["q_ref"] = data_mhe["q"]
    emg_names = ["PectoralisMajorThorax",
                 "BIC",
                 "TRI",
                 "LatissimusDorsi",
                 'TrapeziusScapula_S',
                 #'TrapeziusClavicle',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                  'DeltoideusScapula_P']
    if "P11" in parameters_file_path:
        emg_names.pop(emg_names.index("LatissimusDorsi"))
    key_to_export.append("tau")
    from math import ceil
    track_idx = get_tracking_idx(bio_model, emg_names)
    key_to_plot = ["tau"]
    init_segments = ["Clavicle", "Clavicle",
                "Scapula", "Scapula", "Scapula",
                "Humerus", "Humerus", "Humerus",
                "Forearm"]
    metrics = ["Torque (N.m)"]
    init_joints_names = ["Pro/retraction", "Depression/Elevation",
                    "Pro/retraction", "Lateral/medial rotation", "Tilt",
                    "Plane of elevation", "Elevation", "Axial rotation",
                    "Flexion/extension"]
    for key in key_to_plot:
        n_key = dic_merged[key].shape[0] if key != "tau" else dic_merged[key].shape[0] - 1
        plt.figure(key)
        t = np.linspace(0, 100, 100)
        count = 0
        line_style = "-"
        for i in range(n_key):
            plt.subplot(ceil(n_key / 3), 3, i +1)
            if key in ["u_est", "mus_tau"]:
                key_tmp = ["tau", key] if key != "u_est" else ["muscles_target", key]
                color_tmp = [color] if len(key_tmp) == 1 else ["r", color]
                line_style = ["-", "-"]
            elif key == "tau":
                if color == "g":
                    key_tmp = ["tau", "tau_est", "mus_tau"]
                    color_tmp = ["r", color, color]
                    line_style = ["-", "--", "-"]
                else:
                    key_tmp = ["tau_est", "mus_tau"]
                    color_tmp =  [color, color]
                    line_style = ["--", "-"]
            elif (key == "q_est" or key=="dq_est") and compare_to_fd:
                first_key = "q_ref" if key == "q_est" else "q_dot_ref"
                key_tmp = [first_key, key]
                color_tmp = [color] if len(key_tmp) == 1 else ["r", color]
                line_style = ["-", "-"]
            else:
                key_tmp = [key]
                color_tmp = [color] if len(key_tmp) == 1 else ["r", color]
                line_style = ["-"]
            for idx_j, j in enumerate(key_tmp):
                if j == "muscles_target" and i in track_idx:
                    plt.fill_between(t,
                                     np.mean(dic_merged["cycles"][j], axis=0)[track_idx.index(i), :] - np.std(dic_merged["cycles"][j],
                                                                                             axis=0)[track_idx.index(i), :],
                                     np.mean(dic_merged["cycles"][j], axis=0)[track_idx.index(i), :] + np.std(dic_merged["cycles"][j],
                                                                                             axis=0)[track_idx.index(i), :], alpha=0.3,
                                     color=color_tmp[idx_j])
                    plt.plot(t, np.mean(dic_merged["cycles"][j], axis=0)[track_idx.index(i), :], color=color_tmp[idx_j], alpha=0.7)
                elif j != "muscles_target":
                    plt.fill_between(t,
                            np.mean(dic_merged["cycles"][j], axis=0)[i, :] - np.std(dic_merged["cycles"][j], axis=0)[i, :],
                            np.mean(dic_merged["cycles"][j], axis=0)[i, :] + np.std(dic_merged["cycles"][j], axis=0)[i, :], alpha=0.3, color=color_tmp[idx_j])
                    plt.plot(t, np.mean(dic_merged["cycles"][j], axis=0)[i, :], color=color_tmp[idx_j], alpha=0.7, ls=line_style[idx_j])
                    plt.margins(x=0)
                plt.title(init_segments[i] + " - " + init_joints_names[i], fontsize=20)
                plt.yticks(ticks=plt.yticks()[0], labels=plt.yticks()[0].astype(int))
                if i % 3 == 0:
                    plt.ylabel(metrics[0], fontsize=15)

                if i not in [6, 7, 8]:
                    plt.xticks([])
                    # plt.xticklabels([])
                else:
                    plt.xlabel("Mean cycle (%)", fontsize=15)
                    # ax.tick_params(axis='x', labelsize=font_size - 2)
                # if i == 0:
                #     plt.legend(labels=["ref", "optim", "un"])



if __name__ == '__main__':
    part = "P11"
    participants = ["P10"] #, "P11", "P13", "P14"]
    trials = ["gear_15"]
    cycle = 5
    result_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/results"
    for trial in trials:
        suffix = "test_quad"
        parameters_file_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_gear_20_fd_{cycle}_{suffix}.bio"
        # model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial}_model_scaled_dlc_ribs_new_seth_param_params_id_{cycle}.bioMod"
        # data_path = result_dir + f"/{part}/result_mhe_{trial}_dlc_1_optim_param_True_id_{cycle}_full.bio"
        # plot_all_window(data_path, n_windows=None, plot_by_windows=False, model_path=model)
        model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial}_model_scaled_dlc_ribs_new_seth_param.bioMod"

            #plot_all_window(data_path, n_windows=None, plot_by_windows=False, line_style="--", color = "g", model_path=model)
        #model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial}_model_scaled_dlc_ribs_new_seth_param_params_fd_{cycle}.bioMod"
        file_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
        all_dir = os.listdir(file_dir)
        trial_dir = [dir for dir in all_dir if trial in dir and "result" not in dir][0]
        mhe_file = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial_dir}/result_mhe_torque_driven_{trial}_comparison.bio"
        # plt.show()
        if part != "P16":
            data_path = result_dir + f"/{part}/result_mhe_{trial}_dlc_1_optim_param_True_fd_{cycle}_half_{suffix}.bio"
            plot_cycles(model, data_path, optim_param_path=parameters_file_path, color = "g", mhe_file=mhe_file, compare_to_fd=True)

            #plot_all_window(data_path, n_windows=None, plot_by_windows=False, line_style="--", color = "k", model_path=model)
        # plt.show()
        if part != "P16":
            data_path = result_dir + f"/{part}/result_mhe_{trial}_dlc_1_optim_param_False_1_half_{suffix}.bio"
            plot_cycles(model, data_path, optim_param_path=None, color="b", mhe_file=mhe_file, compare_to_fd=True)
    plt.show()
