from biosiglive import load
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import biorbd
from mhe.utils import apply_params


def get_muscular_torque(x, act, model, parameters_file_path=None):
    """
    Get the muscular torque.
    """
    if parameters_file_path:
        model = apply_params(model, parameters_file_path, with_casadi=False, ratio=True)
    muscular_torque = np.zeros((model.nbQ(), x.shape[1]))
    states = model.stateSet()  # Get the muscle state set
    for i in range(act.shape[1]):
        [states[j].setActivation(act[j, i]) for j in range(act.shape[0])]
        muscular_torque[:, i] = model.muscularJointTorque(
            states, x[: model.nbQ(), i], x[model.nbQ() : model.nbQ() * 2, i]
        ).to_array()
    return muscular_torque

def get_fd_torque(file_path, delta_init=0, final_idx=None, slide_size=1):
    key_to_keep = ["q_ocp", "q_dot_ocp", "tau_ocp"]
    data_mhe = load(file_path)
    data_mhe_tmp = {}
    for key in key_to_keep:
        final_idx = data_mhe[key].shape[-1] if final_idx is None else final_idx
        data_mhe_tmp[key] = data_mhe[key][:, ::slide_size][:, delta_init:final_idx]
    return data_mhe_tmp["q_ocp"], data_mhe_tmp["q_dot_ocp"], data_mhe_tmp["tau_ocp"]


def return_data_from_file(data_path, torque_mhe_file, delta_init=0, final_idx=None):
    result_tmp = load(data_path, merge=True)
    slide_size = int(result_tmp["slide_size"][0])
    q_fd, qdot_fd, tau_fd = get_fd_torque(torque_mhe_file, delta_init=delta_init, final_idx=final_idx, slide_size=slide_size)
    q_fd = q_fd[-biomodel.nbQ():, :]
    qdot_fd = qdot_fd[-biomodel.nbQ():, :]
    tau_fd = tau_fd[-biomodel.nbGeneralizedTorque():, :]
    final_idx = min(tau_fd.shape[1] + delta_init, final_idx) if final_idx is not None else tau_fd.shape[1]
    stat = result_tmp["stat"][delta_init:final_idx]
    final_idx = result_tmp["q_est"].shape[1] if final_idx is None else final_idx
    emg = result_tmp["muscles_target"][:, delta_init:final_idx]
    act = result_tmp["u_est"][:, delta_init:final_idx]
    muscle_tracked = act[result_tmp["muscle_track_idx"][:, 0], :]
    sol_freq = result_tmp["sol_freq"][delta_init:final_idx]
    markers_ref = np.nan_to_num(result_tmp["kin_target"][:, :, delta_init:final_idx])
    q_est = result_tmp["q_est"][:, delta_init:final_idx]
    q_dot_est = result_tmp["dq_est"][:, delta_init:final_idx]
    f_ext = result_tmp["f_ext"][:, delta_init:final_idx]
    tau_res_init = result_tmp["tau_est"][:, delta_init:final_idx]
    markers_est = np.array([biomodel.markers(q_est[:, k]) for k in range(q_est.shape[1])])
    markers_est_array = np.zeros((3, markers_est.shape[1], markers_est.shape[0]))
    for m in range(markers_est.shape[0]):
        markers_est_array[:, :, m] = np.array([mark.to_array() for mark in markers_est[m]]).T
    # import bioviz
    # b = bioviz.Viz(model_path=model)
    # b.load_movement(q_est)
    # b.load_experimental_markers(markers_ref)
    # b.exec()
    #q_est = np.nan_to_num(q_est)
    plt.figure("q_est")
    plt.plot(q_est[8, :], label="q_est")
    plt.plot(q_fd[8, :], label="q_fd")
    plt.figure("q_dot_est")
    plt.plot(q_dot_est[8, :], label="q_dot_est")
    plt.plot(qdot_fd[8, :], label="qdot_fd")
    plt.figure("act")
    plt.plot(muscle_tracked[0, :], label="act")
    plt.plot(act[15, :], label="act")
    plt.plot(emg[0, :], label="emg")
    act_test = np.ones_like(act) * 0.1
    mus_tau = get_muscular_torque(
        np.concatenate((q_est, q_dot_est), axis=0),
        np.clip(act, 0.010001, 0.9999999),
        biomodel,
        parameters_file_path=parameters_file_path,
    )
    mus_tau_test = get_muscular_torque(
        np.concatenate((q_fd, qdot_fd), axis=0),
        np.clip(act_test, 0.010001, 0.9999999),
        biomodel,
        #parameters_file_path=parameters_file_path,
    )
    plt.figure("tau")
    plt.plot(tau_fd[8, :], label="tau_fd")
    plt.plot(mus_tau[8, :] + tau_res_init[8, :], label="mus_tau")
    plt.plot(tau_res_init[8, :], label="tau_res_init")
    plt.plot(mus_tau_test[8, :], label="mus_tau_test")
    #plt.show()
    return result_tmp, emg, act, muscle_tracked, stat, sol_freq, q_est, q_dot_est, f_ext, tau_res_init, markers_ref, markers_est_array, mus_tau, tau_fd


def get_pearson_coef(data, ref_data):
    data_init = data.copy()
    data = data[np.where(np.isfinite(data_init))]
    ref_data = ref_data[np.where(np.isfinite(data_init))]
    return scipy.stats.pearsonr(data, ref_data)[0]

def get_corelation(data, ref_data):
    coor_mat = np.zeros_like(data[..., 0])
    mat_size = len(data.shape)
    if mat_size == 2:
        for i in range(data.shape[0]):
            coor_mat[i] = get_pearson_coef(data[i, :], ref_data[i, :])
        return coor_mat
    elif mat_size == 3:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                coor_mat[i, j] = get_pearson_coef(data[i, j, :], ref_data[i, j, :])
        # delete the nan values
        coor_mat = np.delete(coor_mat, np.where(np.isnan(coor_mat)), axis=1)
        return np.nanmean(coor_mat, axis=0)
    else:
        return None

if __name__ == "__main__":
    delta_init, final_idx = 10, 9000
    participants = [f"P{i}" for i in range(11, 17)]
    init_trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    result_dir = f"/home/mickaelbegon/Documents/Amedeo/results_optim_params"
    idx = 0
    optim = [True, False]
    non_conv = []
    all_iter_full = []
    cycles = ["1", "", "1", "2", "3", "4"]
    count = 0
    print(
        r"""
         Model & \multicolumn{3}{c}{Markers tracking (mm)}& \multicolumn{3}{c}{EMG tracking (\%)} & \multicolumn{3}{c}{Joint torque (N.m)} & \multicolumn{2}{c}{Reserve Torque (\%)}  & Frequency (Hz) \\
              & RMSE & SD & r² & RMSE & SD & r² & RMSE & SD & r² & Mean & SD & Mean & SD  \\             
              \hline
"""
    )
    for c, cycle in enumerate(cycles):
        if cycle== "":
            prefix = f"Ucal"
        proportion_non_conv = []
        freq_total = []
        rmse_mark = None
        rmse_tau = None
        rmse_emg = None
        mean_tau = None
        coor_emg = None
        coor_mark = None
        coor_tau = None
        for p, part in enumerate(participants):
            result_dir = f"/home/mickaelbegon/Documents/Amedeo/results_optim_params/{part}"
            result_dir = f"/mnt/shared/Projet_hand_bike_markerless/optim_params/results/{part}"
            pool_mark_mat = None
            pool_tau_mat = None
            pool_emg_mat = None
            pool_res_tau = None
            pool_mus_tau = None
            coor_emg_mat = None
            coor_mark_mat = None
            coor_tau_mat = None
            nb_non_conv = []
            freq = []
            for t, trial in enumerate(init_trials[p]):
                suffix = "_params_fd" if cycle != "" else ""
                #model_name = f"{trial}_model_scaled_dlc_technical_marker_params_static_root_new_bounds.bioMod" #{suffix}.bioMod"
                model_name = f"{trial}_model_scaled_dlc_technical_marker_params_static_root_new_bounds_static_root.bioMod" #{suffix}.bioMod"
                model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/output_models/{model_name}"
                file_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
                all_dir = os.listdir(file_dir)
                trial_dir = [dir for dir in all_dir if trial in dir and "result" not in dir][0]
                mhe_file = f"/mnt/shared/Projet_hand_bike_markerless/optim_params/reference_data/{part}/reference_torque_{trial}_with_technical_marker_final_optim_param.bio"
                parameters_file_path = None
                if cycle != "":
                    parameters_file_path = f"/mnt/shared/Projet_hand_bike_markerless/optim_params/results_2025-04-14_17-42/{part}/gear_20_n_cycles_{cycle}.bio"

                biomodel = biorbd.Model(model)
                data_path = (
                    result_dir
                    + os.sep
                    + f"result_mhe_{trial}_optim_param_{cycle != ''}_cycle_{cycle}_mhe_0_08_int_1_w4.bio"
                )
                if not os.path.exists(data_path):
                    print(data_path, "not found")
                    continue

                (result_tmp, emg, act, muscle_tracked, stat, sol_freq, q_est, q_dot_est,
                 f_ext, tau_res_init, markers_ref, markers_est_array, mus_tau, tau_fd) = return_data_from_file(data_path, mhe_file, delta_init, final_idx)
                nb_non_conv.extend([(len(stat) - stat.count(0)) * 100 / len(stat)])
                freq.extend(sol_freq)
                error_tau = tau_fd - (mus_tau+ tau_res_init)
                pool_tau_mat = np.concatenate((pool_tau_mat, error_tau), axis=-1) if pool_tau_mat is not None else error_tau
                error_mark = (markers_ref - markers_est_array).mean(axis=0)
                pool_mark_mat = np.concatenate((pool_mark_mat, error_mark), axis=-1) if pool_mark_mat is not None else error_mark
                error_emg = emg - muscle_tracked
                pool_emg_mat = np.concatenate((pool_emg_mat, error_emg), axis=-1) if pool_emg_mat is not None else error_emg
                pool_res_tau = np.concatenate((pool_res_tau, tau_res_init), axis=-1) if pool_res_tau is not None else tau_res_init
                coor_emg_tmp = get_corelation(muscle_tracked, emg)
                coor_emg_mat = np.concatenate((coor_emg_mat, coor_emg_tmp), axis=-1) if coor_emg_mat is not None else coor_emg_tmp
                coor_mark_tmp = get_corelation(markers_est_array, markers_ref)
                coor_mark_mat = np.concatenate((coor_mark_mat, coor_mark_tmp), axis=-1) if coor_mark_mat is not None else coor_mark_tmp
                coor_tau_tmp = get_corelation(mus_tau + tau_res_init, tau_fd)
                coor_tau_mat = np.concatenate((coor_tau_mat, coor_tau_tmp), axis=-1) if coor_tau_mat is not None else coor_tau_tmp

            rmse_mark_tmp = np.sqrt(np.nanmean(pool_mark_mat ** 2, axis=-1))
            rmse_mark = np.concatenate((rmse_mark, rmse_mark_tmp), axis=-1) if rmse_mark is not None else rmse_mark_tmp
            rmse_emg_tmp = np.sqrt(np.nanmean(pool_emg_mat ** 2, axis=-1))
            rmse_emg = np.concatenate((rmse_emg, rmse_emg_tmp), axis=-1) if rmse_emg is not None else rmse_emg_tmp
            rmse_tau_tmp = np.sqrt(np.nanmean(pool_tau_mat[:3] ** 2, axis=-1))
            rmse_tau = np.concatenate((rmse_tau, rmse_tau_tmp), axis=-1) if rmse_tau is not None else rmse_tau_tmp
            mean_tau_tmp = np.nanmean(np.abs(pool_tau_mat[:3]), axis=-1)
            mean_tau = np.concatenate((mean_tau, mean_tau_tmp), axis=-1) if mean_tau is not None else mean_tau_tmp
            coor_emg = np.concatenate((coor_emg, coor_emg_mat), axis=-1) if coor_emg is not None else coor_emg_mat
            coor_mark = np.concatenate((coor_mark, coor_mark_mat), axis=-1) if coor_mark is not None else coor_mark_mat
            coor_tau = np.concatenate((coor_tau, coor_tau_mat), axis=-1) if coor_tau is not None else coor_tau_mat
            freq_total.extend([np.mean(freq)])
            proportion_non_conv.extend([np.mean(nb_non_conv)])

        before = ""
        #b_i = "" if cycle != -1 else r"\textbf{"
        #b_e = "" if cycle != -1 else r"}"
        #if d == 0 and opt is True:

        before = f"$Cal_{cycle}$" if cycle != "" else f"$UCal$"
        #dy_to_print = dy if opt is True else "N/A"
        b_i, b_e = "", ""
        rmse_mark *= 1000
        rmse_emg *= 100
        all_data_tab = [
            np.mean(rmse_mark), np.std(rmse_mark), np.nanmean(coor_mark),
            np.mean(rmse_emg), np.mean(rmse_tau), np.nanmean(coor_emg),
            np.mean(rmse_tau), np.std(rmse_tau), np.nanmean(coor_tau),
            np.mean(mean_tau), np.std(mean_tau),
            np.mean(freq_total), np.mean(proportion_non_conv)
        ]

        print(
            before + f"&"
            f" {b_i}{all_data_tab[0]:0,.2f}{b_e} & {b_i}{all_data_tab[1]:0,.2f}{b_e} & {b_i}{all_data_tab[2]:0,.2f}{b_e} &"
            f" {b_i}{all_data_tab[3]:0,.2f}{b_e} & {b_i}{all_data_tab[4]:0,.2f}{b_e} & {b_i}{all_data_tab[5]:0,.2f}{b_e} &"
            f" {b_i}{all_data_tab[6]:0,.2f}{b_e} & {b_i}{all_data_tab[7]:0,.2f}{b_e} & {b_i}{all_data_tab[8]:0,.2f}{b_e} &"
            f" {b_i}{all_data_tab[9]:0,.2f}{b_e} & {b_i}{all_data_tab[10]:0,.2f}{b_e} & "
            f" {b_i}{all_data_tab[11]:0,.2f}{b_e} & {b_i}{all_data_tab[12]:0,.2f}{b_e}" + r"\\"
        )
