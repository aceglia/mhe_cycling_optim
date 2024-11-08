import bioviz
from biosiglive import load
from mhe.utils import apply_params, load_mhe_results
from optim_params.file_io_utils import get_all_file

if __name__ == '__main__':
    # data = load_mhe_results("results/P10/result_mhe_gear_5_dlc_optim_param_False.bio", 0)
    participants = [f"P{i}" for i in range(10, 17)]
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/reference_data"
    model_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/"

    files, part = get_all_file(participants, data_dir, to_include=["reference_torque_gear_20"])
    data= load(files[0])
    q = data["q_ocp"]
    model = model_dir + f"/P10/output_models/gear_20_model_scaled_dlc_ribs_new_seth_param.bioMod"
    b = bioviz.Viz(model)
    b.load_movement(q)
    b.exec()