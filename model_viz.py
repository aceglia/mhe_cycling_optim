import bioviz
from biosiglive import load
from mhe.utils import apply_params, load_mhe_results

if __name__ == '__main__':
    data = load_mhe_results("results/P10/result_mhe_gear_5_dlc_optim_param_False.bio", 0)
    prefix = "/mnt/shared"
    part = "P10"
    trial = "gear_20"
    model_dir = prefix + f"/Projet_hand_bike_markerless/RGBD/{part}/models"
    parameters_file_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/P10/result_optim_param.bio"
    model = f"{model_dir}/{trial}_model_scaled_dlc_ribs_new_seth_static_root.bioMod"
    b = bioviz.Viz(model)
    b.load_movement(data["q_est"])
    b.exec()