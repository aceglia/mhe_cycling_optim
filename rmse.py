from fontTools.ttLib.tables.otTables import DeltaSetIndexMap

from biosiglive import load
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as st
import biorbd
from mhe.utils import apply_params


def get_muscular_torque(x, act, model, parameters_file_path):
    """
    Get the muscular torque.
    """
    if parameters_file_path:
        model = apply_params(model, parameters_file_path, with_casadi=False, ratio=True)
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

def get_fd_torque(file_path, delta_init=0, final_idx=None):
    key_to_keep = ["q_ocp", "q_dot_ocp", "tau_ocp"]
    data_mhe = load(file_path)
    data_mhe_tmp = {}
    for key in key_to_keep:
        final_idx = data_mhe[key].shape[-1] if final_idx is None else final_idx
        data_mhe_tmp[key] = data_mhe[key][:, delta_init:final_idx]
    return data_mhe_tmp["tau_ocp"]


def return_data_from_file(data_path, torque_mhe_file, delta_init=0, final_idx=None):
    tau_fd = get_fd_torque(torque_mhe_file, delta_init=delta_init, final_idx=final_idx)
    final_idx = min(tau_fd.shape[1] + delta_init, final_idx) if final_idx is not None else tau_fd.shape[1]
    result_tmp = load(data_path, merge=True)
    final_idx = result_tmp["q_est"].shape[1] if final_idx is None else final_idx
    emg = result_tmp["muscles_target"][:, delta_init:final_idx]
    act = result_tmp["u_est"][:, delta_init:final_idx]
    muscle_tracked = act[result_tmp["muscle_track_idx"][:, 0], :]
    stat = result_tmp["stat"][delta_init:final_idx]
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
    q_est = np.nan_to_num(q_est)
    mus_tau = get_muscular_torque(
        np.concatenate((q_est, q_dot_est), axis=0),
        np.clip(act, 0.00001, 0.9999999),
        biomodel,
        parameters_file_path=parameters_file_path,
    )
    return result_tmp, emg, act, muscle_tracked, stat, sol_freq, q_est, q_dot_est, f_ext, tau_res_init, markers_ref, markers_est_array, mus_tau, tau_fd


if __name__ == "__main__":
    delta_init, final_idx = 10, 10000
    participants = [f"P{i}" for i in range(10, 17)]
    init_trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    result_dir = f"/home/mickaelbegon/Documents/Amedeo/results_optim_params"
    idx = 0
    mean_tau = np.zeros((len(init_trials[0]), len(participants)))
    std_tau = np.zeros((len(init_trials[0]), len(participants)))
    rmse_mark = np.zeros((len(init_trials[0]), len(participants)))
    rmse_tau = np.zeros((3, len(init_trials[0]), len(participants)))
    rmse_emg = np.zeros((len(init_trials[0]), len(participants)))
    std_mark = np.zeros((len(init_trials[0]), len(participants)))
    std_emg = np.zeros((len(init_trials[0]), len(participants)))
    rs_emg = np.zeros((len(init_trials[0]), len(participants)))
    rs_mark = np.zeros((len(init_trials[0]), len(participants)))
    rs_tau = np.zeros((len(init_trials[0]), len(participants)))
    all_tau_error = []
    all_mark_error = []
    all_emg_error = []
    optim = [True, False]
    non_conv = []
    all_iter_full = []
    cycles = ["", "1", "2", "3", "4"]
    have_converged = [[]]

    cycle_tau_error = []
    cycle_mark_error = []
    cycle_emg_error = []
    final_error = np.zeros((4, len(cycles), 3))
    non_onv_iter = np.zeros((len(cycles)))
    total_nb_iter = np.zeros((len(cycles)))
    all_freq = np.zeros((2, len(cycles)))

    count = 0

    print(
        r"""
         Model & \multicolumn{3}{c}{Markers tracking (mm)}& \multicolumn{3}{c}{EMG tracking (\%)} & \multicolumn{3}{c}{Joint torque (N.m)} & \multicolumn{2}{c}{Reserve Torque (\%)}  & Frequency (Hz) \\
              & RMSE & SD & r² & RMSE & SD & r² & RMSE & SD & r² & Mean & SD & Mean & SD  \\             
              \hline
"""
    )

    #for c, cycle in enumerate(n_cycle):
    for c, cycle in enumerate(cycles):
        if cycle== "":
            prefix = f"Ucal"
        all_iter = 0
        freq = 0
        std_freq = 0
        proportion_non_conv = 0
        rmse_mark = None
        rmse_tau = None
        rmse_emg = None
        mean_tau = None
        for p, part in enumerate(participants):
            result_dir = f"/home/mickaelbegon/Documents/Amedeo/results_optim_params/{part}"
            pool_mark_mat = None
            pool_tau_mat = None
            pool_emg_mat = None
            pool_res_tau = None
            nb_non_conv = 0

            for t, trial in enumerate(init_trials[p]):
                if part == "P10" and trial == "gear_5":
                    continue
                model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/output_models/{trial}_model_scaled_dlc_technical_marker_params_static_root.bioMod"
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

                nb_non_conv += (len(stat) - stat.count(0)) * 100 / len(stat)
                #all_iter += act[0, :].shape[0]
                freq += float(np.mean(sol_freq))
                std_freq += float(np.std(sol_freq))
                tau_tot = np.array([max(tau) for tau in np.abs(mus_tau + tau_res_init)])
                tau_res = (
                    np.abs(tau_res_init[:-1, :])
                    / np.repeat(tau_tot[:-1, None], tau_res_init.shape[1], axis=1)
                    * 100
                )
                tau_res = tau_res_init

                rmse_tau[0, t, p] = np.sqrt(
                    np.mean(((tau_fd[..., :] - (mus_tau + tau_res_init)) ** 2), axis=1) #* 100 / tau_tot
                ).mean()
                rmse_tau[1, t, p] = np.mean(
                    np.std(tau_fd[..., :] - (mus_tau + tau_res_init), axis=1) #* 100 / tau_tot
                )
                # rmse_tau[0, t, p] = np.sqrt(np.mean(((tau_id[..., :] - (mus_tau + tau_res_init)) ** 2), axis=1)).mean()
                # rmse_tau[1, t, p] = np.mean(np.std(tau_id[..., :] - (mus_tau + tau_res_init), axis=1))
                # plt.plot(tau_id[-2, :], c="r")
                # plt.plot(data_mhe["tau"][-2, delta_init:-delta_final], c="g")
                # plt.plot((mus_tau + tau_res_init)[-2, :])
                # plt.figure("q")
                # plt.plot(data_mhe["q"][-2, delta_init:-delta_final], c="r")
                # plt.plot(q_est[-2, :data_mhe["tau"].shape[1]])
                # plt.figure("qdot")
                # plt.plot(data_mhe["qdot"][-2, delta_init:-delta_final], c="r")
                # plt.plot(q_dot_est[-2, :])
                # plt.show()

                mean_tau[t, p] = np.median(tau_res[3:, ...], axis=1).mean()
                # mean_tau[t, p] = np.sqrt(np.mean(tau_res ** 2, axis=1)).mean()
                std_tau[t, p] = np.std(tau_res[3:, ...], axis=1).mean()
                rmse_mark[t, p] = np.mean(
                    np.sqrt(np.mean(((markers_ref * 1000 - markers_est_array * 1000) ** 2), axis=0)), axis=1
                ).mean()
                std_mark[t, p] = np.mean(
                    np.std(np.mean((markers_ref * 1000 - markers_est_array * 1000), axis=0), axis=1)
                )
                rmse_emg[t, p] = np.sqrt(np.mean(((emg - muscle_tracked) ** 2), axis=1)).mean() * 100
                std_emg[t, p] = np.mean(np.std((emg - muscle_tracked), axis=1)) * 100
                for i in range(emg.shape[0]):
                    corr_matrix = np.corrcoef(emg[i, :], muscle_tracked[i, :])
                    rs_emg[t, p] += corr_matrix[0, 1] ** 2
                rs_emg[t, p] /= emg.shape[0]

                for i in range(tau_fd.shape[0] - 1):
                    corr_matrix = np.corrcoef(tau_fd[i, :], (mus_tau + tau_res_init)[i, :])
                    rmse_tau[2, t, p] += corr_matrix[0, 1] ** 2
                rmse_tau[2, t, p] /= tau_fd.shape[0]
                if rmse_tau[2, t, p] > 1:
                    pass

                for i in range(0, markers_est_array.shape[1]):
                    corr_matrix = np.corrcoef(
                        np.mean(markers_ref[:, i, :], axis=0) * 1000,
                        np.mean(markers_est_array[:, i, :], axis=0) * 1000,
                    )
                    rs_mark[t, p] += corr_matrix[0, 1] ** 2
                rs_mark[t, p] /= markers_ref.shape[1]

        rmse_mark, std_mark = np.nan_to_num(rmse_mark), np.nan_to_num(std_mark)
        rmse_emg, std_emg = np.nan_to_num(rmse_emg), np.nan_to_num(std_emg)
        mean_tau, std_tau = np.nan_to_num(mean_tau), np.nan_to_num(std_tau)
        rmse_tau = np.nan_to_num(rmse_tau)
        rs_mark = np.nan_to_num(rs_mark)
        rs_emg = np.nan_to_num(rs_emg)
        final_error[0, c, :] = np.round(
            [np.mean(rmse_mark[rmse_mark != 0]), np.mean(std_mark[std_mark != 0]), np.mean(rs_mark[rs_mark != 0])],
            2,
        )
        final_error[2, c, :2] = np.round([np.mean(mean_tau[mean_tau != 0]), np.mean(std_tau[std_tau != 0])], 2)
        final_error[1, c, :] = np.round(
            [np.mean(rmse_emg[rmse_emg != 0]), np.mean(std_emg[std_emg != 0]), np.mean(rs_emg[rs_emg != 0])], 2
        )
        final_error[3, c, :] = np.round(
            [
                np.mean(rmse_tau[0, ...][rmse_tau[0, ...] != 0]),
                np.mean(rmse_tau[1, ...][rmse_tau[1, ...] != 0]),
                np.mean(rmse_tau[2, ...][rmse_tau[2, ...] != 0]),
            ],
            2,
        )

        non_onv_iter[c] = nb_non_conv
        all_freq[0, c] = freq / (len(participants) * len(init_trials[0]))
        all_freq[1, c] = std_freq / (len(participants) * len(init_trials[0]))

        before = ""
        #b_i = "" if cycle != -1 else r"\textbf{"
        #b_e = "" if cycle != -1 else r"}"
        #if d == 0 and opt is True:

        before = f"$Cal_{cycle}$" if cycle != "" else f"$UCal$"
        #dy_to_print = dy if opt is True else "N/A"
        b_i, b_e = "", ""

        print(
            prefix + before + f"&"
            f" {b_i}{final_error[ 0, c, 0]:0,.2f}{b_e} & {b_i}{final_error[0, c, 1]:0,.2f}{b_e} & {b_i}{final_error[0, c, 2]:0,.2f}{b_e} &"
            f" {b_i}{final_error[ 1, c, 0]:0,.2f}{b_e} & {b_i}{final_error[1, c, 1]:0,.2f}{b_e} & {b_i}{final_error[1, c, 2]:0,.2f}{b_e} &"
            f" {b_i}{final_error[ 3, c, 0]:0,.2f}{b_e} & {b_i}{final_error[3, c, 1]:0,.2f}{b_e} & {b_i}{final_error[3, c, 2]:0,.2f}{b_e} &"
            f" {b_i}{final_error[ 2, c, 0]:0,.2f}{b_e} & {b_i}{final_error[2, c, 1]:0,.2f}{b_e} & "
            f" {b_i}{all_freq[0,  c]:0,.2f}{b_e} & {b_i}{all_freq[1, c]:0,.2f}{b_e}" + r"\\"
        )

        # all_mark_error.append(np.round([np.mean(rmse_mark[rmse_mark != 0]), np.mean(std_mark[std_mark != 0]), np.mean(rs_mark[rs_mark != 0])], 2))
        # all_tau_error.append(np.round([np.mean(mean_tau[mean_tau != 0]), np.mean(std_tau[std_tau != 0])], 2))
        # all_emg_error.append(np.round([np.mean(rmse_emg[rmse_emg != 0]), np.mean(std_emg[std_emg != 0]), np.mean(rs_emg[rs_emg != 0])], 2))
        # non_conv.append(nb_non_conv)
        # all_iter_full.append(all_iter)
        count += 1

# print("MARKERS(mm) : ", "ID/optim:", all_mark_error[0], "FD/optim:", all_mark_error[1], "none:", all_mark_error[2], )
# print("Tau(N.m): ", "ID/optim:", all_tau_error[0], "FD/optim:", all_tau_error[1], "none:", all_tau_error[2])
# print("emg (%): ", "ID/optim:", all_emg_error[0], "FD/optim:", all_emg_error[1], "none:", all_emg_error[2])
# print("Non converged (%):", "ID", (non_conv[0] / all_iter_full[0]) * 100, "FD", (non_conv[1] / all_iter_full[1]) * 100,
#       "none", (non_conv[2] / all_iter_full[2]) * 100)
