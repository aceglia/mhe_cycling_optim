import matplotlib.pyplot as plt
import numpy as np
from biosiglive import load
from biosiglive import OfflineProcessing, MskFunctions, InverseKinematicsMethods

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


def get_muscular_torque(x, act, model):
    """
    Get the muscular torque.
    """
    muscular_torque = np.zeros((model.nbQ(), x.shape[1]))
    states = model.stateSet()  # Get the muscle state set
    for i in range(act.shape[1]):
        for a, state in zip(act[:, i], states):
            state.setActivation(a)  # And fill it with the current value
        muscular_torque[:, i] = model.muscularJointTorque(
            states, x[: model.nbQ(), i], x[model.nbQ() : model.nbQ() * 2, i]
        ).to_array()
    return muscular_torque


def get_id_torque(x, model, f_ext):
    q = x[: model.nbQ(), :]
    qdot = x[model.nbQ(): model.nbQ() * 2, :]
    qddot = x[model.nbQ() * 2: model.nbQ() * 3, :]
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

def plot_all_window(data_path, n_windows=None, plot_by_windows=False, line_style="-", color="b", model_path=None):
    data = load(data_path, merge=False)
    key_to_plot = ["q_est", "dq_est", "u_est", "muscle_force", "tau_est", "f_ext"]
    ref_key_to_plot = ["q_ref", "dq_ref", "muscles_target", None, None, "f_ext_ref"]
    initial_guess_key = ["q_init", "dq_init", "u_init", None, "tau_init", "f_ext_init"]
    kin_target = None
    target = []
    n_mhe = data[0]["n_mhe"]
    n_q = data[0]["x_init"]["q"].shape[0]

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
    tau = get_id_torque(np.concatenate((q, qdot, qddot), axis=0), msk.model, data_to_plot[0]["f_ext"])
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


if __name__ == '__main__':
    part = "P11"
    model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/gear_20_model_scaled_dlc_ribs_new_seth.bioMod"
    data_path = f"results/{part}/result_mhe_gear_20_dlc_1_optim_param_True.bio"
    plot_all_window(data_path, n_windows=None, plot_by_windows=False, model_path=model)
    model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/gear_20_model_scaled_dlc_ribs_new_seth.bioMod"
    # plt.show()
    data_path = f"results/{part}/result_mhe_gear_20_dlc_1_optim_param_False.bio"
    plot_all_window(data_path, n_windows=None, plot_by_windows=False, line_style="--", color = "g", model_path=model)
    plt.show()
