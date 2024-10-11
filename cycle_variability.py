import numpy as np
from biosiglive import load
import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    muscle_names = ['TrapeziusScapula_M',  # 0
                   'TrapeziusScapula_S',  # 1
                   'TrapeziusScapula_I',  # 2
                   'Rhomboideus_S',  # 3
                   'Rhomboideus_I',  # 3
                   'LevatorScapulae',  # 4
                   'PectoralisMinor',  # 5
                   'TrapeziusClavicle_S',  # 6
                   'SerratusAnterior_I',  # 7
                   'SerratusAnterior_M',  # 7
                   'SerratusAnterior_S',  # 7
                   'Coracobrachialis',  # 8
                   'DeltoideusScapula_P',  # 9
                   'DeltoideusScapula_M',  # 10
                   'TeresMajor',  # 11
                   'Infraspinatus_I',  # 12
                   'Infraspinatus_S',  # 12
                   'TeresMinor',  # 13
                   'Subscapularis_S',  # 14
                   'Subscapularis_M',  # 14
                   'Subscapularis_I',  # 14
                   'Supraspinatus_P',  # 15
                   'Supraspinatus_A',  # 15
                   'DeltoideusClavicle_A',  # 16
                   'PectoralisMajorClavicle_S',  # 17
                   'LatissimusDorsi_S',  # 18
                   'LatissimusDorsi_M',  # 18
                   'LatissimusDorsi_I',  # 18
                   'PectoralisMajorThorax_I',  # 19
                   'PectoralisMajorThorax_M',  # 19
                   # "BRD",
                   # "PT",
                   # "PQ"
                   'TRI_long',  # 20
                   'TRI_lat',  # 20
                   'TRI_med',  # 20
                   'BIC_long',  # 21
                   'BIC_brevis', ]  # 21

    participants = [f"P{i}" for i in range(10, 17)]
    participants.pop(participants.index("P12"))
    participants.pop(participants.index("P15"))
    participants.pop(participants.index("P16"))
    #participants = ["P10", "P11"]
    all_params = np.zeros((len(participants), 2, 34))
    all_params_id = np.zeros((len(participants), 2, 34))
    trials = ["gear_20"]
    nodes = [1, 2, 3, 4, 5, 6]
    result_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/results"
    all_nb_iter = np.zeros((34, len(nodes), 2))
    all_t = np.zeros((34, len(nodes), 2))
    #all_f_iso = np.zeros((34, len(nodes), 2))
    #all_l_optim = np.zeros((34, len(nodes), 2))
    all_track_emg = np.zeros((34, len(nodes), 2))
    all_track_tau = np.zeros((34, len(nodes), 2))

    n_batch = 8
    all_p_iso = np.zeros((len(nodes),len(muscle_names), len(participants)*n_batch))
    all_l_optim = np.zeros((len(nodes), len(muscle_names), len(participants)*n_batch))
    pd_p_iso = pd.DataFrame()
    pd_p_iso["muscles"] = sum([[muscle_names[i]] * n_batch for i in range(len(muscle_names))], []) * len(nodes)
    pd_p_iso["cycles"] = sum([[f"cycle_{i}"] * n_batch * len(muscle_names) for i in range(len(nodes))], [])
    count = 0
    p_iso_all_node = np.zeros((len(muscle_names) * n_batch * len(nodes)))
    all_sd_p = np.zeros((2, len(participants), len(nodes),  len(muscle_names)))
    all_sd_l = np.zeros((2, len(participants), len(nodes),  len(muscle_names)))
    all_time_mhe = []
    solving_time = np.zeros((2, len(participants), len(nodes)))
    converge = np.zeros((len(participants), len(nodes)))
    to_remove_at_the_end = []
    for p, part in enumerate(participants):
        p_iso_all_node = np.zeros((len(muscle_names) * n_batch * len(nodes)))
        count = 0
        file_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
        all_dir = os.listdir(file_dir)
        trials = [dir for dir in all_dir if "gear_20" in dir and "result" not in dir][0]
        mhe_file = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trials}/result_mhe_torque_driven_gear_20.bio"
        mhe_data = load(mhe_file)
        total_time =  mhe_data["total_time_mhe"]
        total_iter = mhe_data["q"].shape[1]
        time_per_h = np.round(total_time / total_iter, 2)
        all_time_mhe.append(time_per_h)
        for t, node in enumerate(nodes):
            param_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_gear_20_fd_{node}_test_quad.bio"
            data_tmp = load(param_path, merge=False)
            print(part, t)
            n_conv = [data_tmp[i]["solver_out"]["status"] for i in range(n_batch)]
            idx_non_conv = np.where(np.array(n_conv) == False)[0]
            n_batch_tmp = n_batch - len(idx_non_conv)
            data_tmp = [data_tmp[i] for i in range(len(data_tmp)) if i not in idx_non_conv]
            if len(data_tmp) == 0:
                to_remove_at_the_end = [[p, t]]
                continue
            converge[p, t] = int(n_conv.count(True) * 100 / n_batch)
            p_iso = []
            _ = [p_iso.append(np.array(data_tmp[i]["p"][0]).tolist()) for i in range(n_batch_tmp)]
            final_list_p = []
            for n in range(len(muscle_names)):
                final_list_p.append([p_iso[i][n] for i in range(n_batch_tmp)])
            l_optim = []
            _ = [l_optim.append(np.array(data_tmp[i]["p"][1]).tolist()) for i in range(n_batch_tmp)]
            final_list_l = []
            for n in range(len(muscle_names)):
                final_list_l.append([l_optim[i][n] for i in range(n_batch_tmp)])
            time = []
            solving_time[0, p, t] = np.mean([data_tmp[i]["solving_time"] for i in range(len(data_tmp))])
            solving_time[1, p, t] = np.std([data_tmp[i]["solving_time"] for i in range(len(data_tmp))])
            #median_f_iso = sum(np.median(np.array(final_list), axis=1).tolist(), [])
            p_iso_all_node[count:count+len(muscle_names)*n_batch_tmp] = np.array(final_list_p).reshape(len(muscle_names)*n_batch_tmp)
            count += len(muscle_names)*n_batch_tmp
            all_sd_p[1, p, t, :] = np.std(np.array(final_list_p).reshape(len(muscle_names), n_batch_tmp), axis=1)
            all_sd_l[1, p, t, :] = np.std(np.array(final_list_l).reshape(len(muscle_names), n_batch_tmp), axis=1)
            all_sd_p[0, p, t, :] = np.median(np.array(final_list_p).reshape(len(muscle_names), n_batch_tmp), axis=1)
            all_sd_l[0, p, t, :] = np.median(np.array(final_list_l).reshape(len(muscle_names), n_batch_tmp), axis=1)

            # l_optim = []
            # _ = [l_optim.append(np.array(data_tmp[i]["p"][1]).tolist()) for i in range(n_batch)]
            # l_optim = np.array(l_optim).reshape(n_batch, len(muscle_names),).T
            # #for b in range(n_batch):
            # all_p_iso[t, :, count:count+n_batch] = p_iso
            # all_l_optim[t, :, count:count+n_batch] = l_optim
            #
            # count += n_batch
    mat_per_node = []
    value_per_tab = np.zeros((2, 4, len(nodes)))
    print(r"""
    \begin{table}[]
        \centering
        \begin{tabular}{l c c c c c c c c c}
        \hline
             Cycles & SD & Time (s) & Convergence (\%) & MHE solving frequency (Hz) & Total time (min)  \\
             \hline
             """)
    for i in range(len(nodes)):
        b_i = "" if i + 1 != -1 else r"\textbf{"
        b_e = "" if i + 1 != -1 else r"}"
        value_per_tab[0, 1, i] = np.mean(solving_time[0, :, i][solving_time[0, :, i]!=0])
        value_per_tab[1, 1, i] = np.mean(solving_time[1, :, i][solving_time[1, :, i]!=0])
        for k in range(2):
            value_param_p = all_sd_p[k].mean(axis=-1)
            value_param_l = all_sd_l[k].mean(axis=-1)
            value_per_tab[k, 0, i] = (np.mean(value_param_p[:, i][value_param_p[:, i]!=0]) + np.mean(value_param_l[:, i][value_param_l[:, i]!=0]))/2
        value_per_tab[0, 3, i] = np.mean(all_time_mhe)
        value_per_tab[0, 2, i] = np.mean(converge, axis=0)[i]
        last_column = "" if i != 0 else " \multirow{6}*{" + f"{1/value_per_tab[0, 3, i]:0,.2f}" + "}"
        print(f"{b_i}{i+1}{b_e} & {b_i}{value_per_tab[1, 0, i]:0,.2f}{b_e}  & {b_i}{value_per_tab[0, 1, i]:0,.2f}{b_e}   &"
              f" {b_i}{int(value_per_tab[0, 2, i])}{b_e} & {last_column} "
              f" & {(value_per_tab[0, 3, i] * (i + 1) * 15 + value_per_tab[0, 1, i]):0,.2f}" + r"\\")
    print("\hline")



        #pd_p_iso["p"] = p_iso_all_node
        #sn.boxplot(data=pd_p_iso, x="muscles", y="p", hue="cycles")
        #plt.show()
    # print(r"""
    # \begin{table}[]
    #     \centering
    #     \begin{tabular}{l c c c c c c c c}
    #     \hline
    #          Cycles & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8   \\
    #          \hline
    #          """
    #          f"Mean $\pm$ SD & {value_per_tab[0, 0, 0]:0,.2f} $\pm$  {value_per_tab[1, 0, 0]:0,.2f} & {value_per_tab[0, 0, 0]:0,.2f} "
    #       f"$\pm${value_per_tab[1, 0, 1]:0,.2f}  & {value_per_tab[0, 0, 2]:0,.2f} $\pm$ {value_per_tab[1, 0, 2]:0,.2f}  & {value_per_tab[0, 0, 3]:0,.2f}"
    #       f"  $\pm$ {value_per_tab[1, 0, 3]:0,.2f} "
    #       f"& {value_per_tab[0, 0, 4]:0,.2f}  $\pm$ {value_per_tab[1, 0, 4]:0,.2f} & {value_per_tab[0, 0, 5]:0,.2f} $\pm$ {value_per_tab[1, 0, 5]:0,.2f} "
    #       f"& {value_per_tab[0, 0, 6]:0,.2f}  $\pm$ {value_per_tab[1, 0, 6]:0,.2f} & {value_per_tab[0, 0, 7]:0,.2f} $\pm$ {value_per_tab[1, 0, 7]:0,.2f} "
    #       + r"\\" +" \n"
    #          f"Time (s) & {value_per_tab[0, 1, 0]:0,.2f}  & {value_per_tab[0, 1, 1]:0,.2f}  & {value_per_tab[0, 1, 2]:0,.2f} "
    #                f" & {value_per_tab[0, 1, 3]:0,.2f}  & {value_per_tab[0, 1, 4]:0,.2f}  & {value_per_tab[0, 1, 5]:0,.2f}  "
    #                f"& {value_per_tab[0, 1, 6]:0,.2f}  & {value_per_tab[0, 1, 7]:0,.2f} " + r"\\" +" \n"
    #          f"Convergence rate & {value_per_tab[0, 2, 0]} &  {value_per_tab[0, 2, 1]} &  {value_per_tab[0, 2, 2]} &  {value_per_tab[0, 2, 3]} &"
    #          f"  {value_per_tab[0, 2, 4]} &  {value_per_tab[0, 2, 5]} &  {value_per_tab[0, 2, 6]} &  {value_per_tab[0, 2, 7]}" + r"\\" +" \n"
    #          r"MHE time / Horizon (s) & \multicolumn{8}{c}{" + f"{value_per_tab[0, 3, 0]:0,.2f}" + r"} \\" +" \n"
    #          r"""\hline
    #     \end{tabular}
    #     \caption{Caption}
    #     \label{tab:my_label}
    # \end{table}""")
    pass