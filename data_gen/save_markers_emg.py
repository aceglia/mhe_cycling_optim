from pyomeca import Markers, Analogs
import matplotlib.pyplot as plt
import numpy as np
from biosiglive import save
import os
import glob
from pathlib import Path
import biorbd
import pickle


def save_results(sol, c3d_file_path):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """
    data = dict(
        values=sol.values
    )
    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    value = data_tmp["values"]

    return value


def compute_mvc(
        nb_muscles: int,
        mvc_trials: np.ndarray,
        window_size: int,
) -> list:
    """    Compute MVC from several mvc_trials.

    Parameters
    ----------
    nb_muscles : int
        Number of muscles.
    mvc_trials : numpy.ndarray
        EMG data for all trials.
    window_size : int
       Size of the window to compute MVC. Usually it is 1 second so the data rate.
    tmp_file : str
        Name of the temporary file.
    output_file : str
        Name of the output file.
    save_file : bool
        If true, save the results.

    Returns
    -------
    list
        MVC for each muscle.

    """
    mvc_list_max = []
    for i in range(nb_muscles):
        mvc_temp = -np.sort(-mvc_trials, axis=1)
        if i == 0:
            mvc_list_max = mvc_temp[:, :window_size]
        else:
            mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :window_size]), axis=1)
    mvc_list_max = -np.sort(-mvc_list_max, axis=1)[:, :window_size]
    mvc_list_max = np.median(mvc_list_max, axis=1)

    return mvc_list_max



# datas = Markers.from_c3d("/home/lim/Documents/Stage_Antoine/Antoine_Leroy/fichier_c3d/pedalage_1.c3d", usecols=['STER', 'XIPH', 'C7', 'T10', 'CLAV_SC', 'CLAV_AC', 'Acrom', 'SCAP_AA', 'SCAp_IA', 'DELT', 'EPIC_L', 'EPIC_M', 'ARMl', 'LARM_Elb', 'STYL_R', 'STYL_u', 'hand1', 'hand2', 'hand3', 'hand4'])
# # print(datas)
#
# EMG = Analogs.from_c3d("/home/lim/Documents/Stage_Antoine/Antoine_Leroy/fichier_c3d/pedalage_1.c3d", usecols=['subscap.IM EMG1', 'infraspin.IM EMG2', 'supspin.IM EMG3', 'pecmaj.IM EMG4', 'trap_sup.IM EMG5', 'Trap_med.IM EMG6', 'trap_inf.IM EMG7', 'bic.IM EMG8', 'tri.IM EMG9', 'lat.IM EMG10', 'delt_ant.IM EMG11', 'delt_med.IM EMG12', 'delt_post.IM EMG13', 'Pec_2.IM EMG14'])
#
# print(EMG)

channel_1 = ['subscap.IM EMG1', 'infraspin.IM EMG2', 'supspin.IM EMG3', 'pecmaj.IM EMG4', 'trap_sup.IM EMG5', 'Trap_med.IM EMG6', 'trap_inf.IM EMG7', 'bic.IM EMG8', 'tri.IM EMG9', 'lat.IM EMG10', 'delt_ant.IM EMG11', 'delt_med.IM EMG12', 'delt_post.IM EMG13', 'Pec_2.IM EMG14']
channel_2 = ['Sensor 1.IM EMG1', 'Sensor 2.IM EMG2', 'Sensor 3.IM EMG3', 'Sensor 4.IM EMG4', 'Sensor 5.IM EMG5', 'Sensor 6.IM EMG6', 'Sensor 7.IM EMG7', 'Sensor 8.IM EMG8', 'Sensor 9.IM EMG9', 'Sensor 10.IM EMG10', 'Sensor 11.IM EMG11', 'Sensor 12.IM EMG12', 'Sensor 13.IM EMG13', 'Sensor 14.IM EMG14']
channel_3 = ['STER', 'XIPH', 'C7', 'T10', 'CLAV_SC', 'CLAV_AC', 'Acrom', 'SCAP_AA', 'SCAp_IA', 'DELT', 'EPIC_L', 'EPIC_M', 'ARMl', 'LARM_Elb', 'STYL_R', 'STYL_u', 'hand1', 'hand2', 'hand3', 'hand4']

emg_proc = []
nb_muscles = 14
data_dir = "/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/data_gen/fichier_c3d"
c3d_files = glob.glob(data_dir + "/**.c3d")
for file in c3d_files:
    if "MVC" in file:
        emg_tmp = Analogs.from_c3d(file, usecols=channel_2)
    else:
        emg_tmp = Analogs.from_c3d(file, usecols=channel_1)
    emg_processed = (
        emg_tmp.meca.band_pass(order=2, cutoff=[10, 425])
        .meca.center()
        .meca.abs()
        .meca.low_pass(order=4, cutoff=5, freq=emg_tmp.rate)
    )
    if len(emg_proc) == 0:
        emg_proc = emg_processed
    else:
        emg_proc = np.append(emg_proc, emg_processed, axis=1)

mvc_windows = 2000
mvc_list_max = np.ndarray((nb_muscles, mvc_windows))
output_file = "mvc.mat"
mvc = compute_mvc(nb_muscles, emg_proc, mvc_windows)
file = "/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/data_gen/fichier_c3d/pedalage_4.c3d"
datas_markers = Markers.from_c3d(file, usecols=channel_3).values
datas_markers = datas_markers[:3, :16, :]*0.001
idx = np.argwhere(np.isnan(datas_markers))
for i in range(idx.shape[0]):
    idx_coordonnees = idx[i][0]
    idx_marker = idx[i][1]
    idx_nan = idx[i][2]
    datas_markers[idx_coordonnees, idx_marker, idx_nan] = datas_markers[
        idx_coordonnees, idx_marker, idx_nan - 1]

emg_pedalage = Analogs.from_c3d(file, usecols=channel_1)
emg_processed_2 = (
        emg_pedalage.meca.band_pass(order=2, cutoff=[10, 425])
        .meca.center()
        .meca.abs()
        .meca.low_pass(order=4, cutoff=5, freq=emg_pedalage.rate)
    )
emg_processed_2 = emg_processed_2[:, ::20]

for i in range(len(emg_processed_2[1])):
    emg_processed_2[:, i] = np.divide(emg_processed_2[:, i], mvc)

save({"markers": datas_markers, "emg_proc": emg_processed_2[:13, :].values}, f"saves/{Path(file).stem}_proc")
