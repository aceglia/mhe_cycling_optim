from biosiglive import load, save
import os
import shutil


if __name__ == '__main__':
    participants = [f"P{i}" for i in range(10, 17)]
    init_trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    cycles = ["", "1", "2", "3", "4"]
    result_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/results"
    for p, part in enumerate(participants):
        result_dir_part = os.path.join(result_dir, part)
        for t, trial in enumerate(init_trials[p]):
            for c in cycles:
                optim = "True" if c != "" else "False"
                data_path = (
                        result_dir_part
                        + os.sep
                        + f"result_mhe_{trial}_optim_param_{optim}_cycle_{c}_mhe_0_1_int_1_w3.bio"
                )
                data = load(data_path)
                new_data_path = data_path.replace(result_dir_part, f"/home/mickaelbegon/Documents/Amedeo/results_optim_params/{part}")
                save(data, new_data_path.replace(".bio", "_merged.bio"), safe=False)
                # copy file
                #new_data_path = data_path.replace(result_dir_part, f"/home/mickaelbegon/Documents/Amedeo/results_optim_params/{part}")
                #shutil.copy2(data_path, new_data_path)
                print("file", data_path, "merged")
