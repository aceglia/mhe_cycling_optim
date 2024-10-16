from optim_params.generate_data import TorqueEstimator
from optim_params.file_io_utils import get_experimental_data, get_all_file

prefix = "/mnt/shared"

def get_ocp_weights():
    weights = {
        "q": 100,
        "qdot": 1000,
        "tau": 100,
        "q_prev": 10,
        "qdot_prev": 1000,
        "f_ext":1000000000,
        "markers":100000000,
    }
    return weights



if __name__ == '__main__':
    with_mhe = True
    data_rate = 60
    final_time = 0.1
    n_shooting = int(final_time * data_rate)
    participants = [f"P{i}" for i in range(10, 17)]
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/RGBD"
    trials = ["gear_5", "gear_10", "gear_15", "gear_20"]
    # model_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial_short}_model_scaled_{source[:-2]}_ribs_new_seth_param.bioMod"
    files, participants = get_all_file(participants, data_dir, trial_names=trials, to_include=["gear"], to_exclude=["less", "more", "result"])
    for file, participant in zip(files, participants):
        trial_short = file.split("/")[-1].split("_")[0] + "_" + file.split("/")[-1].split("_")[1]
        file_path = prefix + f"/Projet_hand_bike_markerless/process_data/{participant}" + f"/result_biomech_{trial_short}_normal_500_down_b1.bio"
        torque_estimator = TorqueEstimator()
        torque_estimator.init_experimental_data(get_experimental_data(file_path, source="dlc_1", downsample=2, n_stop=1000))
        biorbd_model_path = (f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{participant}/models/{trial_short}_model_scaled_dlc_ribs_new_seth_param.bioMod")
        torque_estimator.init_ocp(biorbd_model_path,
                                  final_time,
                                  n_shooting,
                                  with_external_loads=True,
                                  use_mhe=with_mhe,
                                  track_previous=False,
                                  weights=get_ocp_weights()
                                  )
        torque_estimator.compute_torque(from_direct_dynamics=True, use_residuals=True, save_data=False, output_path=None)
        break