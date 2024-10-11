
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

if __name__ == '__main__':
    delta_init, delta_final = 10, 10
    participants = [f"P{i}" for i in range(10, 15)]
    if "P12" in participants:
        participants.pop(participants.index("P12"))
    # participants.pop(participants.index("P15"))
    # participants.pop(participants.index("P16"))
    all_params = np.zeros((len(participants), 2, 35))
    init_trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    #init_trials = [["gear_10"]] * len(participants)

    result_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/results"
    idx = 0
    mean_tau = np.zeros((len(init_trials[0]), len(participants)))
    std_tau = np.zeros((len(init_trials[0]), len(participants)))
    rmse_mark = np.zeros((len(init_trials[0]), len(participants)))
    rmse_tau= np.zeros((3, len(init_trials[0]), len(participants)))
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
    n_cycle = [1, 2, 3, 4, 5, 6] #1, 2, 3, 4, 5]
    dyn = ["fd"]
    len_d = len(dyn) + 1
    have_converged = [[]] * len_d

    cycle_tau_error = []
    cycle_mark_error = []
    cycle_emg_error = []
    final_error = np.zeros((len_d, 4, len(n_cycle), 3))
    non_onv_iter = np.zeros((len_d, len(n_cycle)))
    total_nb_iter = np.zeros((len_d, len(n_cycle)))
    all_freq = np.zeros((2, len_d, len(n_cycle)))


    count = 0

    print(r"""
         Model & \multicolumn{3}{c}{Markers tracking (mm)}& \multicolumn{3}{c}{EMG tracking (\%)} & \multicolumn{3}{c}{Joint torque (N.m)} & \multicolumn{2}{c}{Reserve Torque (\%)}  & Frequency (Hz) \\
              & RMSE & SD & r² & RMSE & SD & r² & RMSE & SD & r² & Mean & SD & Mean & SD  \\             
              \hline
""")

    for c, cycle in enumerate(n_cycle):
        if cycle == n_cycle[-1]:
            dyn.append("")
        for d, dy in enumerate(dyn):
            opt = dy != ""
            prefix = ""
            if opt is False and cycle != n_cycle[-1]:
                continue
            elif opt is False and cycle == n_cycle[-1]:
                prefix = f"Ucal"
                cycle = 1
            if d == 0 and c == 0 and opt is True:
                prefix = ""
            nb_non_conv = 0
            all_iter = 0
            freq = 0
            std_freq = 0
            for p,  part in enumerate(participants):
                result_dir = f"/mnt/shared/Projet_hand_bike_markerless/optim_params/results/{part}"
                for t, trial in enumerate(init_trials[p]):
                    if opt:
                        model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial}_model_scaled_dlc_ribs_new_seth_param_params_{dy}_{cycle}.bioMod"
                    else:
                        model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial}_model_scaled_dlc_ribs_new_seth_param.bioMod"
                    model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial}_model_scaled_dlc_ribs_new_seth_param.bioMod"
                    file_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
                    all_dir = os.listdir(file_dir)
                    trial_dir = [dir for dir in all_dir if trial in dir and "result" not in dir][0]
                    mhe_file = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial_dir}/result_mhe_torque_driven_{trial}_comparison.bio"
                    suffix = "test_quad"
                    parameters_file_path = None
                    if opt is True:
                        parameters_file_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_gear_20_fd_{cycle}_{suffix}.bio"
                    data_mhe = None
                    if mhe_file:
                        key_to_keep = ["q", "qdot", "tau"]
                        data_mhe = load(mhe_file)
                        data_mhe_tmp = {}
                        for key in key_to_keep:
                            data_mhe_tmp[key] = data_mhe[key]
                    else:
                        raise RuntimeError("No mhe file provided")

                    biomodel = biorbd.Model(model)
                    file_dy = f"_{dy}" if dy != "" else ""

                    data_path = result_dir + os.sep + f"result_mhe_{trial}_dlc_1_optim_param_{str(opt)}{file_dy}_{cycle}_half_{suffix}.bio"
                    if not os.path.exists(data_path):
                        print(data_path, "not found")
                        continue
                    result_tmp = load(data_path, merge=False)
                    emg = np.array([k["muscles_target"][:, idx] for k in result_tmp][delta_init:-delta_final]).T
                    act = np.array([k["u_est"][:, idx] for k in result_tmp][delta_init:-delta_final]).T
                    stat = [k["stat"] for k in result_tmp][delta_init:-delta_final]
                    sol_freq = [k["sol_freq"] for k in result_tmp][1:]
                    nb_non_conv += np.argwhere(np.isnan(act[0, :])).shape[0]
                    all_iter += act[0, :].shape[0]
                    freq += float(np.mean(sol_freq))
                    std_freq += float(np.std(sol_freq))
                    # if np.argwhere(np.isnan(act)).shape[0] != 0:
                    #     have_converged[count].append(False)
                    #act = np.nan_to_num(act)
                    muscle_track_idx = result_tmp[0]["muscle_track_idx"]
                    muscle_tracked = act.copy()[muscle_track_idx, :]
                    markers_ref = np.nan_to_num(np.array([k["kin_target"][:, :, idx] for k in result_tmp][delta_init:-delta_final]).T)
                    markers_ref = np.swapaxes(markers_ref, 0, 1)
                    q_est = np.array([k["q_est"][:, idx] for k in result_tmp][delta_init:-delta_final]).T
                    q_dot_est = np.array([k["dq_est"][:, idx] for k in result_tmp][delta_init:-delta_final]).T
                    f_ext = np.array([k["f_ext"][:, idx] for k in result_tmp][delta_init:-delta_final]).T
                    markers_est = np.array([biomodel.markers(q_est[:, k]) for k in range(q_est.shape[1])])
                    markers_est_array = np.zeros((3, markers_est.shape[1], markers_est.shape[0]))
                    for m in range(markers_est.shape[0]):
                        markers_est_array[:, :, m] = np.array([mark.to_array() for mark in markers_est[m]]).T
                    #markers_est_array = np.nan_to_num(markers_est_array)
                    q_est = np.nan_to_num(np.array([k["q_est"][:, idx] for k in result_tmp][delta_init:-delta_final]).T)
                    # mus_tau = get_muscular_torque(np.concatenate((q_est, q_dot_est), axis=0), np.clip(act, 0.00001, 0.9999999), biomodel)
                    tau_res_init = np.array([k["tau_est"][:, idx] for k in result_tmp][delta_init:-delta_final]).T
                    # total_tau = np.abs(np.array([max(tau) for tau in mus_tau + tau_res]))
                    # tau_res = np.abs(tau_res) * 100 / np.repeat(total_tau[:, None], tau_res.shape[1], axis=1)
                    mus_tau = get_muscular_torque(np.concatenate((q_est, q_dot_est), axis=0),
                                                                np.clip(act, 0.00001, 0.9999999), biomodel,
                                                                parameters_file_path=parameters_file_path)
                    tau_id = get_id_torque(q_est, q_dot_est, biomodel, f_ext, rate=60)
                    tai_id = data_mhe_tmp["tau"][delta_init:-delta_final]

                    tau_tot = np.array([max(tau) for tau in np.abs(mus_tau + tau_res_init)])
                    tau_res = np.abs(tau_res_init[:-1, :]) / np.repeat(tau_tot[:-1, None], tau_res_init.shape[1], axis=1) * 100
                    rmse_tau[0, t, p] = np.sqrt(np.mean(((tau_id[..., :] - (mus_tau + tau_res_init)) ** 2), axis=1) * 100 / tau_tot).mean()
                    rmse_tau[1, t, p] = np.mean(np.std(tau_id[..., :] - (mus_tau + tau_res_init), axis=1) * 100 / tau_tot)
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


                    mean_tau[t, p] = np.mean(tau_res, axis=1).mean()
                    mean_tau[t, p] = np.median(tau_res, axis=1).mean()
                    #mean_tau[t, p] = np.sqrt(np.mean(tau_res ** 2, axis=1)).mean()
                    std_tau[t, p] = np.std(tau_res, axis=1).mean()
                    rmse_mark[t, p] = np.mean(np.sqrt(np.mean(((markers_ref * 1000 - markers_est_array * 1000) ** 2), axis=0)), axis=1).mean()
                    std_mark[t, p] = np.mean(np.std(np.mean((markers_ref * 1000 - markers_est_array * 1000), axis=0), axis=1))
                    rmse_emg[t, p] = np.sqrt(np.mean(((emg - muscle_tracked) ** 2), axis=1)).mean() * 100
                    std_emg[t, p] = np.mean(np.std((emg - muscle_tracked), axis=1)) * 100
                    for i in range(emg.shape[0]):
                        corr_matrix = np.corrcoef(emg[i, :], muscle_tracked[i, :])
                        rs_emg[t, p] += corr_matrix[0, 1] ** 2
                    rs_emg[t, p] /= emg.shape[0]

                    for i in range(tau_id.shape[0] - 1):
                        corr_matrix = np.corrcoef(tau_id[i, :], (mus_tau + tau_res_init)[i, :])
                        rmse_tau[2, t, p] += corr_matrix[0, 1] ** 2
                    rmse_tau[2, t, p] /= tau_id.shape[0]
                    if rmse_tau[2, t, p] > 1:
                        pass

                    for i in range(3, markers_est_array.shape[1]):
                        corr_matrix = np.corrcoef(np.mean(markers_ref[:, i, :], axis=0) * 1000, np.mean(markers_est_array[:, i, :], axis=0) * 1000)
                        rs_mark[t, p] += corr_matrix[0, 1] ** 2
                    rs_mark[t, p] /= markers_ref.shape[1]-3

            rmse_mark, std_mark = np.nan_to_num(rmse_mark), np.nan_to_num(std_mark)
            rmse_emg, std_emg = np.nan_to_num(rmse_emg), np.nan_to_num(std_emg)
            mean_tau, std_tau = np.nan_to_num(mean_tau), np.nan_to_num(std_tau)
            rmse_tau = np.nan_to_num(rmse_tau)
            rs_mark = np.nan_to_num(rs_mark)
            rs_emg = np.nan_to_num(rs_emg)
            final_error[d, 0, c, :] = np.round([np.mean(rmse_mark[rmse_mark != 0]), np.mean(std_mark[std_mark != 0]), np.mean(rs_mark[rs_mark != 0])], 2)
            final_error[d, 2, c, :2] = np.round([np.mean(mean_tau[mean_tau != 0]), np.mean(std_tau[std_tau != 0])], 2)
            final_error[d, 1, c, :] = np.round([np.mean(rmse_emg[rmse_emg != 0]), np.mean(std_emg[std_emg != 0]), np.mean(rs_emg[rs_emg != 0])], 2)
            final_error[d, 3, c, :] = np.round([np.mean(rmse_tau[0, ...][rmse_tau[0, ...] != 0]), np.mean(rmse_tau[1, ...][rmse_tau[1, ...] != 0]), np.mean(rmse_tau[2, ...][rmse_tau[2, ...] != 0])], 2)

            non_onv_iter[d, c] = nb_non_conv
            total_nb_iter[d, c] = all_iter
            all_freq[0, d, c] = freq / (len(participants) * len(init_trials[0]))
            all_freq[1, d, c] = std_freq / (len(participants) * len(init_trials[0]))

            before = ""
            b_i = "" if cycle != -1 else r"\textbf{"
            b_e = "" if cycle != -1 else r"}"
            if d == 0 and opt is True:
                before = f'{b_i}$Cal_{cycle}${b_e}'
            elif d == 2 and opt is False:
                before = "N/A"
            dy_to_print = dy if opt is True else "N/A"

            print(
                prefix + before + f'&'
                                  f' {b_i}{final_error[d, 0, c, 0]:0,.2f}{b_e} & {b_i}{final_error[d, 0, c, 1]:0,.2f}{b_e} & {b_i}{final_error[d, 0, c, 2]:0,.2f}{b_e} &'
                                  f' {b_i}{final_error[d, 1, c, 0]:0,.2f}{b_e} & {b_i}{final_error[d, 1, c, 1]:0,.2f}{b_e} & {b_i}{final_error[d, 1, c, 2]:0,.2f}{b_e} &'
                                  f' {b_i}{final_error[d, 3, c, 0]:0,.2f}{b_e} & {b_i}{final_error[d, 3, c, 1]:0,.2f}{b_e} & {b_i}{final_error[d, 3, c, 2]:0,.2f}{b_e} &'
                                  f' {b_i}{final_error[d, 2, c, 0]:0,.2f}{b_e} & {b_i}{final_error[d, 2, c, 1]:0,.2f}{b_e} & '
                                  f' {b_i}{all_freq[0, d, c]:0,.2f}{b_e} & {b_i}{all_freq[1, d, c]:0,.2f}{b_e}' + r"\\")


            # all_mark_error.append(np.round([np.mean(rmse_mark[rmse_mark != 0]), np.mean(std_mark[std_mark != 0]), np.mean(rs_mark[rs_mark != 0])], 2))
            # all_tau_error.append(np.round([np.mean(mean_tau[mean_tau != 0]), np.mean(std_tau[std_tau != 0])], 2))
            # all_emg_error.append(np.round([np.mean(rmse_emg[rmse_emg != 0]), np.mean(std_emg[std_emg != 0]), np.mean(rs_emg[rs_emg != 0])], 2))
            #non_conv.append(nb_non_conv)
            #all_iter_full.append(all_iter)
            count += 1

    # print("MARKERS(mm) : ", "ID/optim:", all_mark_error[0], "FD/optim:", all_mark_error[1], "none:", all_mark_error[2], )
    # print("Tau(N.m): ", "ID/optim:", all_tau_error[0], "FD/optim:", all_tau_error[1], "none:", all_tau_error[2])
    # print("emg (%): ", "ID/optim:", all_emg_error[0], "FD/optim:", all_emg_error[1], "none:", all_emg_error[2])
    # print("Non converged (%):", "ID", (non_conv[0] / all_iter_full[0]) * 100, "FD", (non_conv[1] / all_iter_full[1]) * 100,
    #       "none", (non_conv[2] / all_iter_full[2]) * 100)
