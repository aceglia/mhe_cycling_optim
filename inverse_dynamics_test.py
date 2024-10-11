import os
import numpy as np
from biosiglive import load, OfflineProcessing, MskFunctions, InverseKinematicsMethods
import biorbd
import bioviz
import matplotlib.pyplot as plt

if __name__ == '__main__':
    part = "P10"
    file_name = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
    all_dir = os.listdir(file_name)
    trial = [dir for dir in all_dir if "gear_20" in dir][0]

    model = "normal_500_down_b1"
    # trials = [f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial}/reoriented_dlc_markers.bio"]
    prefix = "/mnt/shared"
    trials = [
        prefix + f"/Projet_hand_bike_markerless/process_data/{part}/result_biomech_{trial.split('_')[0]}_{trial.split('_')[1]}_{model}_no_root_offline.bio"]
    source = "dlc_1"
    biorbd_model_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/gear_20_model_scaled_{source[:-2]}_ribs_new_seth_param.bioMod"

    source = "vicon"
    biorbd_model_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/gear_20_model_scaled_{source}_new_seth_param.bioMod"

    for trial in trials:
        kalman_data = load(trial)
        n_start = 300  #int(7) + 8
        n_stop = 600  #int(390) - 157
        f_ext = kalman_data["f_ext"][:, n_start:n_stop]
        q_init = kalman_data[source]["q_raw"][:, n_start:n_stop]
        q_dot_init = kalman_data[source]["q_dot"][:, n_start:n_stop]
        q_ddot_init = kalman_data[source]["q_ddot"][:, n_start:n_stop]
        tau_init = kalman_data[source]["tau"][:, n_start:n_stop]
        model = biorbd.Model(biorbd_model_path)
        names = kalman_data[source]["marker_names"]
        ia_idx = names.index("SCAP_IA")
        ts_idx = names.index("SCAP_TS")
        mark_ia = kalman_data[source][f"tracked_markers"][:, ia_idx, :].copy()
        mark_ts = kalman_data[source][f"tracked_markers"][:, ts_idx, :].copy()
        kalman_data[source][f"tracked_markers"][:, ia_idx, :] = mark_ts
        kalman_data[source][f"tracked_markers"][:, ts_idx, :] = mark_ia
        markers_init = kalman_data[source][f"tracked_markers"][:, :, n_start:n_stop]
        import os

        file_name = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
        all_dir = os.listdir(file_name)
        trial = [dir for dir in all_dir if "gear_20" in dir][0]
        file = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial}/result_{trial.split('/')[-1]}.bio"
        data = load(file)
        results = {}
        results["q_est"] = data["q"][:, :]
        results["dq_est"] = data["qdot"][:, :]
        results["tau_est"] = data["tau"][:, :]
        msk_fun = MskFunctions(biorbd_model_path, markers_init.shape[2], 120)
        q_new, q_dot_new, _ = msk_fun.compute_inverse_kinematics(markers_init, method=InverseKinematicsMethods.BiorbdLeastSquare, qdot_from_finite_difference=True)
        q_filtered = OfflineProcessing().butter_lowpass_filter(q_new,
                6, 120, 2)
        q_filtered = OfflineProcessing().butter_lowpass_filter( q_init,
                6, 120, 2)
        qdot_new = np.zeros_like(q_init)
        for i in range(1, q_filtered.shape[1] - 2):
            qdot_new[:, i] = (q_filtered[:, i + 1] - q_filtered[:, i - 1]) / (1 / 120)

        qddot_new = np.zeros_like(qdot_new)
        for i in range(1, qdot_new.shape[1] - 2):
            qddot_new[:, i] = (qdot_new[:, i + 1] - qdot_new[:, i - 1]) / (1 / 120)

        qddot = np.zeros_like(q_init)
        for i in range(1, results["q_est"].shape[1] - 2):
            qddot[:, i] = (results["dq_est"][:, i + 1] - results["dq_est"][:, i - 1]) / (1 / 120)

        tau = np.zeros_like(q_init)
        tau_bis = np.zeros_like(q_init)
        tau_new = np.zeros_like(q_init)


        for i in range(q_init.shape[1]):
            B = [0, 0, 0, 1]
            all_jcs = model.allGlobalJCS(q_filtered[:, i])
            RT = all_jcs[-1].to_array()
            # A = RT @ A
            B = RT @ B
            vecteur_OB = B[:3]
            f_ext[:3, i] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])
            f_ext[:3, i] = np.zeros_like(f_ext[:3, i])
            f_ext[3:6, i] = f_ext[3:6, i]
            # force_global = change_ref_for_global(ind_1, q, model, force_locale)
            # ddq = nlp.model.forward_dynamics(q, qdot, tau, force_global)
            ext = model.externalForceSet()
            ext.add("hand_left", f_ext[:, i])
            tau[:, i] = model.InverseDynamics(q_init[:, i], q_dot_init[:, i], q_ddot_init[:, i], ext).to_array()
            tau[:, i] = model.InverseDynamics(q_init[:, i], q_dot_init[:, i], q_ddot_init[:, i]).to_array()
            try:
                dq = results["dq_est"][:, i]
                q = results["q_est"][:, i]
            except:
                dq = np.zeros((10))
                q = np.zeros((10))

            tau_bis[:, i] = model.InverseDynamics(q, dq, qddot[:, i]).to_array()
            tau_new[:, i] = model.InverseDynamics(q_filtered[:, i], qdot_new[:, i], qddot_new[:, i], ext).to_array()




        for i in range(markers_init.shape[1]):
            plt.subplot(4, 5, i + 1)
            for j in range(3):
                plt.plot(markers_init[j, i, :])

        factor = 1  # 57.3
        plt.figure("q")
        for i in range(q_init.shape[0]):
            plt.subplot(4, 4, i + 1)
            #plt.plot(q_init[i, :] * 57.3)
            plt.plot(q_filtered[i, :]* 57.3)
            plt.plot(q_new[i, :]* 57.3)


        plt.figure("qdot")
        for i in range(q_dot_init.shape[0]):
            plt.subplot(4, 4, i + 1)
            #plt.plot(q_dot_init[i, :] * factor)
            plt.plot(q_dot_new[i, :] * factor)



        plt.figure("qddot")
        for i in range(q_dot_init.shape[0]):
            plt.subplot(4, 4, i + 1)
            #plt.plot(q_ddot_init[i, :] * factor)
            plt.plot(qddot[i, :] * factor)
            plt.plot(qddot_new[i, :] * factor)


        plt.figure("tau")
        for i in range(tau_init.shape[0]):
            plt.subplot(4, 4, i + 1)
            #plt.plot(tau_init[i, :], "r")
            #plt.plot(tau[i, :])
            plt.plot(tau_bis[i, :])
            plt.plot(tau_new[i, :])


        plt.show()