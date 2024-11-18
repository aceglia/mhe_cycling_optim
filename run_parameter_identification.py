import os
import time

from optim_params.parameters_identifier import ParametersIdentifier
from optim_params.identification_utils import get_all_muscle_len
from optim_params.enum import Parameters
from optim_params.file_io_utils import get_all_file, get_data_dict
import biorbd
import numpy as np


weights = {"tau_tracking": 10,
           "activation_tracking": 100,
           "min_act": 1,
           "min_f_iso": 20,
           "min_lm_optim": 80,
           "min_pas_torque": 10
           }

emg_names_init = ["PectoralisMajorThorax",
             "BIC",
             "TRI",
             "LatissimusDorsi",
             'TrapeziusScapula_S',
             # 'TrapeziusClavicle',
             "DeltoideusClavicle_A",
             'DeltoideusScapula_M',
             'DeltoideusScapula_P']


def update_data(initial_data, random_idx_list):
    cycle_size = initial_data["q"].shape[-1]
    n_cycles = len(random_idx_list)
    q, qdot, tau, f_ext, emg = (initial_data["q"], initial_data["qdot"],
                                     initial_data["tau"],
                                     initial_data["f_ext"], initial_data["emg"])
    q, qdot, tau, f_ext, emg = (q[random_idx_list, ...], qdot[random_idx_list, ...],
                                     tau[random_idx_list, ...], f_ext[random_idx_list, ...],
                                     emg[random_idx_list, ...])
    q_final = np.zeros((q.shape[1], cycle_size * n_cycles))
    qdot_final = np.zeros((qdot.shape[1], cycle_size * n_cycles))
    tau_final = np.zeros((tau.shape[1], cycle_size * n_cycles))
    f_ext_final = np.zeros((f_ext.shape[1], cycle_size  * n_cycles))
    emg_final = np.zeros((emg.shape[1], cycle_size  * n_cycles))
    for i in range(n_cycles):
        q_final[:, i * cycle_size:(i + 1) * cycle_size] = q[i, :, :]
        qdot_final[:, i * cycle_size:(i + 1) * cycle_size] = qdot[i, :, :]
        tau_final[:, i * cycle_size:(i + 1) * cycle_size] = tau[i, :, :]
        f_ext_final[:, i * cycle_size:(i + 1) * cycle_size] = f_ext[i, :, :]
        emg_final[:, i * cycle_size:(i + 1) * cycle_size] = emg[i, :, :]
    import matplotlib.pyplot as plt
    plt.figure("q")
    for i in range(q_final.shape[0]):
        plt.subplot(q_final.shape[0]//3+1, 3, i+1)
        for j in range(q.shape[0]):
            plt.plot(q[j, i, :])
    plt.figure("qdot")
    for i in range(qdot_final.shape[0]):
        plt.subplot(qdot_final.shape[0]//3+1, 3, i+1)
        plt.plot(qdot[0, i, :])
    dict_data = {"q": q_final, "qdot": qdot_final, "tau": tau_final, "f_ext": f_ext_final, "emg": emg_final}
    return dict_data

def initialize_bounds_and_mapping(optim_param_list, biorbd_model_path, q, use_p_mapping=False):
    optim_param_list = [p.value for p in optim_param_list]
    param_bounds = [[0.0, 1.0] for _ in optim_param_list]
    p_init = [1] * len(optim_param_list)
    all_muscle_len = None
    eigen_model = biorbd.Model(biorbd_model_path)
    for p_idx, param in enumerate(optim_param_list):
        if param == "f_iso":
            param_bounds[p_idx] = [0.5, 3]
        elif param == "lm_optim":
            all_muscle_len = get_all_muscle_len(eigen_model, q)
            param_bounds[p_idx] = [0.5, 2]
        elif param == "lt_slack":
            param_bounds[p_idx] = [0.8, 1.2]
        else:
            raise ValueError(f"Parameter {param} not recognized")
    p_mapping = [list(range(eigen_model.nbMuscles())), list(range(eigen_model.nbMuscles()))]
    p_mapping_list = [p_mapping] * len(optim_param_list)
    list_mapping = list(range(eigen_model.nbMuscles()))
    if use_p_mapping and "f_iso" in optim_param_list:
        list_mapping = [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11, 12, 12, 13, 14, 14, 14, 15, 15, 16, 17, 18, 18,
                                         18, 19, 19, 20, 20, 20, 21, 21]
        p_mapping = [list(range(max(list_mapping) + 1)), list_mapping]
        p_mapping_list[optim_param_list.index("f_iso")] = p_mapping
    return param_bounds, p_init, p_mapping_list, all_muscle_len, list_mapping

if __name__ == '__main__':
    with_param = True
    with_residual_torque = True
    use_ratio_tracking = True
    save_data = False
    participants = [f"P{i}" for i in range(10, 17)]
    params_to_optimize = [Parameters.f_iso, Parameters.lm_optim]
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/reference_data"
    model_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/"


    files, part = get_all_file(participants, data_dir, to_include=["reference_torque_gear_20"])
    n_cycles = [3, 1,2,3,4,5]
    batch_size = 3
    date = time.strftime("%Y-%m-%d_%H-%M")
    for file, participant in zip(files, part):
        list_tmp = file.replace(".bio", "").split("/")[-1].split("_")
        trial_short = "gear_" + list_tmp[list_tmp.index("gear") + 1]
        model_path = model_dir + f"/{participant}/output_models/{trial_short}_model_scaled_dlc_ribs_new_seth_param.bioMod"
        emg_names = emg_names_init.copy()
        if participant == 'P11':
            emg_names.pop(emg_names.index('LatissimusDorsi'))
        for n_cycle in n_cycles:
            output_file = f"/mnt/shared/Projet_hand_bike_markerless/optim_params/results_{date}/{participant}"
            if save_data:
                if not os.path.exists(output_file):
                    os.makedirs(output_file)
            output_file = output_file + f"/{trial_short}_n_cycles_{n_cycle}"

            identifier = ParametersIdentifier(params_to_optimize)
            cycle_size = 15
            initial_data, idx_random = get_data_dict(file, n_cycles=n_cycle, batch_size=batch_size, rate=120,
                                                              cycle_size=cycle_size, from_id=False, em_delay=0.0)
            if "min_lm_optim" in weights:
                weights["lm_optim"] = weights["min_lm_optim"] * n_cycle # + 10 * (n_cycles - 1)
            if "min_f_iso" in weights:
                weights["min_f_iso"] = weights["min_f_iso"] * n_cycle #+ 10 * (n_cycles - 1)
            for i in range(batch_size):
                identifier.load_experimental_data(update_data(initial_data,idx_random[i]))
                param_bounds, p_init, p_mapping_list, all_muscle_len, list_mapping = initialize_bounds_and_mapping(
                    params_to_optimize, model_path,
                                              identifier.q, use_p_mapping=False)
                identifier.initialize_problem(model_path, p_mapping_list, with_residual_torques=with_residual_torque,
                                               threads=1, weights=weights, scaling_factor=(1, (0.01, 0.001), 1), emg_names=emg_names,
                                              all_muscle_len=all_muscle_len, l_norm_bounded=False, p_init=None,
                                              param_bounds=param_bounds, use_sx=True, add_muscle_torque_constraint=False)
                identifier.solve(save_results=save_data, output_file=output_file, max_iter=3000,
                                 hessian_approximation="exact",
                                 linear_solver="ma57",
                                 #obj_scaling_factor=0.0001,
                                 print_level=5,
                                 tol=1e-4,
                                 #ma57_pivtol=1e-2,
                                 #derivative_test="first-order",
                                 #mu_strategy="adaptive",
                                 # nlp_scaling_method = "none",
                                 plot=True, objective_scale_factor=10, cycle_number=idx_random[i], batch_number=i)

                print(f"Optimization for participant {participant} and trial {trial_short} is done for batch {i} and cycles {n_cycle}")

