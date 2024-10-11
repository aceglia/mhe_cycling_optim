import os
import biorbd as biorbd_eigen



if __name__ == '__main__':
    muscle_list_wu = [
        "LVS",  # 0
        "TRP2",  # 2 "TRPsup_bis"
        "TRP3",  # 2
        "TRP4",  # 3
        "RMN",  # 4
        "RMJ1",  # 5
        "RMJ2",  # 5
        "SRA1",  # 6
        "SRA2",  # 6
        "SRA3",  # 6
        "PMN",  # 7
        "TRP1",  # 1 "TRPsup"
        "SBCL",  # 8
        "DELT1",  # 9 "DELTant"
        "PECM1",  # 10
        "DELT2",  # 11 "DELTmed"
        "DELT3",  # 12 "DELTpost"
        "SUPSP",  # 13
        "INFSP",  # 14
        "SUBSC",  # 15
        "TMIN",  # 16
        "TMAJ",  # 16
        "CORB",  # 17
        "PECM2",  # 10
        "PECM3",  # 10
        "LAT",  # 18
        "bic_l",  # 19
        "bic_b",  # 19
        "tric_long",  # 20
        "tric_lat",  # 20
        "tric_med", ]  # 20

    muscle_list_seth = ['TrapeziusScapula_M',  # 0
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

    participants = ["P"+ str(i) for i in range(10, 17)]
    for part in participants:
        biorbd_model = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial_short}_model_scaled_dlc_ribs_new_seth_param_with_root.bioMod"
        with open(biorbd_model, "r") as file:
            str_model = file.read()
        param_model = biorbd_eigen.Model(biorbd_model)
        start_idx = 0
        str_model_tmp = ""
        count = 0
        end_idx = 0
        while True:
            idx_tmp = str_model.find("optimalLength", start_idx) + len(r"optimalLength\t") - 1
            if idx_tmp == -1 + len(r"optimalLength\t") - 1:
                str_model_tmp += str_model[end_idx:]
                break
            str_model_tmp += str_model[end_idx:idx_tmp]
            end_idx = str_model.find('\n', idx_tmp)
            str_model_tmp += str(param_model.muscle(count).characteristics().optimalLength())
            start_idx = end_idx
            count += 1

        str_model = str_model_tmp

        start_idx = 0
        str_model_tmp = ""
        count = 0
        end_idx = 0
        while True:
            idx_tmp = str_model.find("maximalForce", start_idx) + len(r"maximalForce\t") - 1
            if idx_tmp == -1 + len(r"maximalForce\t") - 1:
                str_model_tmp += str_model[end_idx:]
                break
            str_model_tmp += str_model[end_idx:idx_tmp]
            end_idx = str_model.find('\n', idx_tmp)
            str_model_tmp += str(param_model.muscle(count).characteristics().forceIsoMax())
            start_idx = end_idx
            count += 1
        str_model = str_model_tmp
        start_idx = 0
        str_model_tmp = ""
        count = 0
        end_idx = 0
        while True:
            idx_tmp = str_model.find("tendonSlackLength", start_idx) + len(r"tendonSlackLength\t") - 1
            if idx_tmp == -1 + len(r"tendonSlackLength\t") - 1:
                str_model_tmp += str_model[end_idx:]
                break
            str_model_tmp += str_model[end_idx:idx_tmp]
            end_idx = str_model.find('\n', idx_tmp)
            str_model_tmp += str(param_model.muscle(count).characteristics().tendonSlackLength())
            start_idx = end_idx
            count += 1
        param_model = None
        new_file_name = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial_short}_model_scaled_dlc_ribs_param_with_root_wu.bioMod"
        #new_file_name = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial_short}_model_scaled_dlc_ribs_param_with_root_seth.bioMod"
        with open(new_file_name, "w") as file:
            file.write(str_model_tmp)