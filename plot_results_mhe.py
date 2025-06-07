import matplotlib.pyplot as plt
import numpy as np
from biosiglive import load
import biorbd
from mhe.utils import apply_params
import os


def plot_results(data_path, model, key_to_plot, key_target=None, optim_param_path=None, color="b"):
    idx_to_export = 0
    data = load(data_path, merge=False)
    muscle_idx = data[0]["muscle_track_idx"]
    dic_merged = {}
    for key in data[0].keys():
        if isinstance(data[0][key], np.ndarray):
            dic_merged[key] = np.array([k[key][..., idx_to_export] for k in data]).T
            if key == "kin_target" and len(dic_merged[key].shape) == 3:
                dic_merged[key] = np.swapaxes(dic_merged[key], 0, 1)
    kin = np.concatenate((dic_merged["q_est"], dic_merged["dq_est"]), axis=0)
    biomodel = biorbd.Model(model)
    import bioviz
    b = bioviz.Viz(model)
    b.load_movement(dic_merged["x_ref"][: biomodel.nbQ(), :])
    b.load_experimental_markers(dic_merged["kin_target"])
    b.exec()

    dic_merged["dq_ref"] = dic_merged["x_ref"][biomodel.nbQ() : biomodel.nbQ() * 2, :]
    dic_merged["q_ref"] = dic_merged["x_ref"][: biomodel.nbQ(), :]

    dic_merged["mus_tau"] = get_muscular_torque(
        kin, dic_merged["u_est"], biomodel, parameters_file_path=optim_param_path
    )
    for k, key in enumerate(key_to_plot):
        if key not in dic_merged.keys():
            print(f"{key} not in data")
            continue
        data_tmp = dic_merged[key]
        target = None
        if key_target is not None and key_target[k] is not None:
            if key_target[k] not in dic_merged.keys():
                print(f"{key_target[k]} not in data")
                continue
            target = dic_merged[key_target[k]]
        plt.figure(key)
        for i in range(data_tmp.shape[0]):
            plt.subplot(int(np.ceil(data_tmp.shape[0] / 4)), 4, i + 1)
            plt.plot(data_tmp[i, :], color=color)
            if target is not None:
                plot_target(i, key_target[k], target, muscle_idx)
            if key == "u_est":
                plt.title(biomodel.muscleNames()[i].to_string())


def plot_target(i, key_target, target, idx_muscle):
    if target is None:
        return
    if key_target == "muscles_target":
        if i in idx_muscle:
            plt.plot(target[idx_muscle.index(i), :], "r")
    else:
        plt.plot(target[i, :], color="r")


def get_muscular_torque(x, act, model, parameters_file_path=None, with_casadi=False, ratio=True):
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


if __name__ == "__main__":
    part = "P11" # , "P11", "P13", "P14"]
    trials = ["gear_5"]
    cycle = 3
    result_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/results"
    for trial in trials:
        suffix = "test_quad"
        parameters_file_path = (
            f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_gear_20_fd_{cycle}_{suffix}.bio"
        )
        model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/output_models/{trial}_model_scaled_dlc_technical_marker_params_static_root.bioMod"
        file_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
        all_dir = os.listdir(file_dir)
        trial_dir = [dir for dir in all_dir if trial in dir and "result" not in dir][0]
        mhe_file = None
        key_to_plot = ["q_est", "dq_est", "u_est", "tau_est", "f_ext", "muscle_force", "mus_tau"]
        key_ref = ["q_ref", "dq_ref", "muscles_target", None, "f_ext_ref", None, None]
        #data_path = result_dir + f"/{part}/result_mhe_{trial}_dlc_1_optim_param_False_track_markers.bio"
        #plot_results(data_path, model, key_to_plot, key_ref, color="b")
        data_path = result_dir + f"/{part}/result_mhe_{trial}_dlc_1_optim_param_True_track_markers.bio"
        plot_results(data_path, model, key_to_plot, key_ref, color="g")
        plt.show()
