from optim_params.generate_data import TorqueEstimator
from optim_params.file_io_utils import get_experimental_data, get_all_file

prefix = "/mnt/shared"


def _compute_new_bounds(data):
    bounds = [
        # "rotations xyz // thorax\n\t\ttranslations xyz // thorax\n\t\t//ranges \n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n",
        "rotations xyz // thorax\n\t\ttranslations xyz // thorax\n",  # thorax
        "rotations x\n\t\tranges\n\t\t\t\t-3 3",  # clavicle
        "rotations y\n\t\tranges\n\t\t\t\t-3 3",  # clavicle
        "//rotations z\n\t\t//ranges\n\t\t\t\t//-3 3",  # clavicle
        # "rotations xyz\n\t\tranges\n\t\t\t\t-0.2 1",  # scapula
        "rotations xyz\n\t\tranges\n\t\t\t\t-3 3\n\t\t\t\t-3 3\n\t\t\t\t-3 3",  # scapula
        "rotations x\n\t\tranges\n\t\t\t\t-3 3",  # shoulder
        "rotations y\n\t\tranges\n\t\t\t\t-3 3",  # shoulder
        "rotations z\n\t\tranges\n\t\t\t\t-3 3",  # shoulder
        "rotations z\n\t\tranges\n\t\t\t\t-0.5 3",  # elbow
        "rotations y\n\t\tranges\n\t\t\t\t-3 3",  # forearm
        "//rotations xy\n\t\t//ranges\n\t\t\t\t//0.1 0.8",  # Hand
    ]
    # bounds = [
    #     "rotations xyz // thorax\n\t\ttranslations xyz // thorax\n\t\tranges \n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n",
    #     "rotations x\n\t\tranges\n\t\t\t\t-0.7 0.8",  # clavicle
    #     "rotations y\n\t\tranges\n\t\t\t\t-0.5 0.5",  # clavicle
    #     "//rotations z\n\t\tranges\n\t\t\t\t-3 3",  # clavicle
    #     # "rotations xyz\n\t\tranges\n\t\t\t\t-0.2 1",  # scapula
    #     "rotations xyz\n\t\tranges\n\t\t\t\t-0.1 1\n\t\t\t\t-0.1 0.8\n\t\t\t\t-0.2 0.5",  # scapula
    #     "rotations x\n\t\tranges\n\t\t\t\t-0.4 0.8",  # shoulder
    #     "rotations y\n\t\tranges\n\t\t\t\t0.2 1",  # shoulder
    #     "rotations z\n\t\tranges\n\t\t\t\t-1.2 0.2",  # shoulder
    #     "rotations z\n\t\tranges\n\t\t\t\t0.8 2.2",  # elbow
    #     "rotations y\n\t\tranges\n\t\t\t\t0.3 0.8",  # forearm
    # ]
    data_tmp = data
    idx_start = 0
    count = 0
    while True:
        if count == 0:
            idx_start = data_tmp.find("rotations xyz // thorax")
            to_replace = "rotations xyz // thorax\n\t\ttranslations xyz // thorax"
            idx_start = idx_start + len(to_replace)
            data_tmp = data_tmp.replace(to_replace, bounds[0])
            count += 1
        else:
            idx_start = data_tmp.find("rotations", idx_start)
            if idx_start == -1:
                break
            idx_end = data_tmp.find("endsegment", idx_start) - 2
            data_tmp = data_tmp.replace(data_tmp[idx_start:idx_end], bounds[count])
            idx_start = idx_end
            count += 1
    return data_tmp


def get_ocp_weights():
    weights = {
        "q": 100,
        "qdot": 1000,
        "tau": 100,
        "q_prev": 10,
        "qdot_prev": 1000,
        "f_ext": 1000000000,
        "markers": 100000000,
    }
    return weights


if __name__ == "__main__":
    with_mhe = True
    data_rate = 120
    final_time = 0.12
    n_shooting = int(final_time * data_rate)
    participants = [f"P{i}" for i in range(14, 15)]
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/RGBD"
    trials = ["gear_20"]  # ["gear_5", "gear_10", "gear_15", "gear_20"]
    # model_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial_short}_model_scaled_{source[:-2]}_ribs_new_seth_param.bioMod"
    files, participants = get_all_file(
        participants, data_dir, trial_names=trials, to_include=["gear_20"], to_exclude=["less", "more", "result"]
    )
    for file, participant in zip(files, participants):
        trial_short = file.split("/")[-1].split("_")[0] + "_" + file.split("/")[-1].split("_")[1]
        # file_path = prefix + f"/Projet_hand_bike_markerless/process_data/{participant}" + f"/result_biomech_{trial_short}_normal_500_down_b1.bio"
        file_path = (
            prefix
            + f"/Projet_hand_bike_markerless/process_data/{participant}"
            + f"/result_biomech_{trial_short}_with_technical_marker_params.bio"
        )
        output_path = (
            prefix
            + f"/Projet_hand_bike_markerless/optim_params/reference_data/{participant}"
            + f"/reference_torque_{trial_short}_with_technical_marker.bio"
        )
        torque_estimator = TorqueEstimator()
        torque_estimator.init_experimental_data(
            get_experimental_data(file_path, source="dlc_1", downsample=1, n_stop=None)
        )
        biorbd_model_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{participant}/output_models/{trial_short}_model_scaled_dlc_technical_marker_params.bioMod"
        # with open(biorbd_model_path, "r") as f:
        #     data = f.read()
        # # data = _compute_new_bounds(data)
        # new_biorbd_model_path = biorbd_model_path.replace(".bioMod", ".bioMod")
        # with open(new_biorbd_model_path, "w") as f:
        #     f.write(data)
        torque_estimator.init_ocp(
            biorbd_model_path,
            final_time,
            n_shooting,
            with_external_loads=True,
            use_mhe=with_mhe,
            track_previous=True,
            weights=get_ocp_weights(),
        )
        torque_estimator.compute_torque(from_direct_dynamics=True, save_data=True, output_path=output_path)
        # break
