import matplotlib.pyplot as plt
import numpy as np
import biorbd
import os
from mhe.utils import apply_params, get_ratio
from biosiglive import load

def compute_muscle_tau(model_path, a, q, qdot, param, param_list):
    model = biorbd.Model(model_path)
    mus_j_torque = np.zeros((q.shape[0], q.shape[1]))
    ratio = None
    plt.figure(f"tau_comparison")
    model_tmp = model
    for j in range(2):
        if j == 0 and "lt_slack" not in param_list:
            ratio = get_ratio(model_tmp, use_casadi=False)
        if j == 1:
            model_tmp = apply_params(model_tmp, params=param, optimized_params=param_list, with_casadi=False, ratio=ratio)
        for k in range(q.shape[1]):
            muscle_states = model_tmp.stateSet()
            for m in range(model_tmp.nbMuscles()):
                muscle_states[m].setActivation(a[m, k])
            mus_j_torque[:, k] = model_tmp.muscularJointTorque(muscle_states, q[:, k], qdot[:, k]).to_array()
    return mus_j_torque


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
    nodes = [1, 2, 3]
    result_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/results"
    all_nb_iter = np.zeros((34, len(nodes), 2))
    all_t = np.zeros((34, len(nodes), 2))
    all_f_iso = np.zeros((34, len(nodes), 2))
    all_l_optim = np.zeros((34, len(nodes), 2))
    all_track_emg = np.zeros((34, len(nodes), 2))
    all_track_tau = np.zeros((34, len(nodes), 2))
    dyn = ["fd", "id"]
    print(r"""
         Part & Number of cycle & Dyn & f iso & l optim & Track emg & Track Tau & Iter & Time \\
             \hline""")
    for p,  part in enumerate(participants):
        for n, trial in enumerate(trials):
            for t, node in enumerate(nodes):
                for d, dy in enumerate(dyn):
                    prefix = "&"
                    if t == 0 and d == 0:
                        prefix = "\multirow{8}*{" + f"P{p}" + " }&"
                    model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial}_model_scaled_dlc_ribs_new_seth_param.bioMod"
                    suffix = "_id" if dy == "id" else ""
                    param_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_{trial}_{dy}_{node}_test.bio"
                    data_tmp = load(param_path)
                    all_f_iso[:, t, d] = np.array(data_tmp["p"][0])[:, 0]
                    all_l_optim[:, t, d] = np.array(data_tmp["p"][1])[:, 0]
                    all_t[:, t, d] = np.repeat(data_tmp["solving_time"], 34)
                    all_nb_iter[:, t, d] = np.repeat(data_tmp["solver_out"]["n_iter"], 34)
                    tracked_act = data_tmp["a"][data_tmp["muscle_track_idx"], :]
                    emg = data_tmp["emg"]
                    mus_tau = compute_muscle_tau(model, data_tmp["a"], data_tmp["q"], data_tmp["qdot"],
                                                 data_tmp["p"], param_list=data_tmp["optimized_params"])
                    tau_tot = mus_tau + data_tmp["pas_tau"]
                    emg_error = np.sqrt(np.mean(((emg - tracked_act) ** 2), axis=1)).mean() * 100
                    tau_error = np.sqrt(np.mean(((data_tmp["tracked_torque"] - mus_tau) ** 2), axis=1)).mean()
                    color = ""
                    if data_tmp["solver_out"]["status"] is True:
                        color = "\cellcolor{green!25}"
                    before = ""
                    if d == 0:
                        before = "\multirow{2}*{" + f'{node}' + "}"
                    print(prefix + before + f'& {dy} & {all_f_iso[:, t, d].mean():0,.2f} & {all_l_optim[:, t, d].mean():0,.2f} & {emg_error:0,.2f} & {tau_error:0,.2f} & {color} {data_tmp["solver_out"]["n_iter"]} & {data_tmp["solving_time"]:0,.2f}' + r"\\")

                    # param_path_id = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_id.bio"
                    # data_tmp_id = load(param_path_id)
                    # all_params_id[p, 0, :], all_params_id[p, 1, :] = np.array(data_tmp_id["p"][0])[:, 0],  np.array(data_tmp_id["p"][1])[:, 0]
