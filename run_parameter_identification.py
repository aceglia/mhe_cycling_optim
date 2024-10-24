from optim_params.parameters_identifier import ParametersIdentifier
from optim_params.identification_utils import get_all_muscle_len
from optim_params.enum import Parameters
from optim_params.file_io_utils import get_all_file, get_data_dict
from biosiglive import load
import biorbd
import numpy as np


weights = {"tau_tracking": 2,
           "activation_tracking": 10,
           "min_act": 1,
           "min_f_iso": 5,
           "min_lm_optim": 5,
           "min_lt_slack": 100,
           "min_pas_torque": 0.6,
           "ratio_tracking": 1,
           "dynamics": 100}
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
    dict_data = {"q": q_final, "qdot": qdot_final, "tau": tau_final, "f_ext": f_ext_final, "emg": emg_final}
    return dict_data

def initialize_bounds_and_mapping(optim_param_list, biorbd_model_path, q, use_p_mapping=False):
    optim_param_list = [p.value for p in optim_param_list]
    param_bounds = [[0, 1] for _ in optim_param_list]
    p_init = [1]
    all_muscle_len = None
    eigen_model = biorbd.Model(biorbd_model_path)
    muscle_list = [m.to_string() for m in eigen_model.muscleNames()]
    for p_idx, param in enumerate(optim_param_list):
        if param == "f_iso":
            param_bounds[p_idx] = [0.5, 2.5]
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
    participants = [f"P{i}" for i in range(9, 17)]
    params_to_optimize = [Parameters.f_iso, Parameters.lm_optim]
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/reference_data"
    model_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/"

    files, part = get_all_file(participants, data_dir, to_include=["reference_torque", "gear_20"])
    batch_size = 2
    for file, participant in zip(files, part):
        list_tmp = file.replace(".bio", "").split("/")[-1].split("_")
        trial_short = "gear_" + list_tmp[list_tmp.index("gear") + 1]
        model_path = model_dir + f"/{participant}/models/{trial_short}_model_scaled_dlc_ribs_new_seth_param.bioMod"
        emg_names = emg_names_init.copy()
        if part == 'P11':
            emg_names.pop(emg_names.index('LatissimusDorsi'))
        identifier = ParametersIdentifier(params_to_optimize)
        initial_data, idx_random = get_data_dict(file, n_cycles=2, batch_size=batch_size, rate=120,
                                                          cycle_size=15, from_id=False)
        for i in range(batch_size):
            identifier.load_experimental_data(update_data(initial_data,idx_random[i]))
            param_bounds, p_init, p_mapping_list, all_muscle_len, list_mapping = initialize_bounds_and_mapping(
                params_to_optimize, model_path,
                                          identifier.q, use_p_mapping=False)
            identifier.initialize_problem(model_path, p_mapping_list, with_residual_torques=with_residual_torque,
                                           threads=6, weights=weights, scaling_factor=(1, (1, 1), 1), emg_names=emg_names,
                                          all_muscle_len=all_muscle_len, l_norm_bounded=False, p_init=p_init,
                                          param_bounds=param_bounds, use_sx=False)
            identifier.solve(save_results=False, output_file=data_dir, max_iter=1000, hessian_approximation="exact",
                             linear_solver="ma57")
            print(f"Optimization for participant {participant} and trial {trial_short} is done for batch {i}")

