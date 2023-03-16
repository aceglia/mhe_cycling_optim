import numpy as np

from biosiglive import save, load
import pickle
from scipy.interpolate import interp1d


def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    value = data_tmp["values"]

    return value


if __name__ == "__main__":

    data_markers = get_created_data_from_pickle("/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/sandbox/cycling/markers_g_pedalage_1.pkl")
    data_markers = data_markers[:3, :, :]*0.001
    idx = np.argwhere(np.isnan(data_markers))
    for i in range (idx):

    # markers_ref = np.zeros((3, data_markers.shape[1], int(data_markers.shape[2])))
    # for i in range(3):
    #     x = np.linspace(0, data_markers.shape[2] / 100, data_markers.shape[2])
    #     f_mark = interp1d(x, data_markers[i, :, :])
    #     x_new = np.linspace(0, data_markers.shape[2] / 100, int(data_markers.shape[2]))
    #     markers_ref[i, :, :] = f_mark(x_new)
    # markers_ref = np.concatenate((markers_ref[:, :15, :], markers_ref[:, 16:, :]), axis=1)
    # markers_ref = np.concatenate((markers_ref[:, :10, :], markers_ref[:, 11:, :]), axis=1)
    # markers_ref = markers_ref[:, 1:, :]
    data_emg = get_created_data_from_pickle("/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/sandbox/cycling/emg_g_pedalage_1.pkl")
    data_emg = data_emg[:13, :]
    data_q = get_created_data_from_pickle("/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/sandbox/cycling/q_recons/q_recons_pedalage_scaled_model.pkl")
    data_qdot = get_created_data_from_pickle("/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/sandbox/cycling/q_recons/q_dot_recons_pedalage_scaled_model.pkl")
    data_kalman = np.concatenate((data_q, data_qdot), axis=0)
    data_loaded = load("trials/data_abd_sans_poid.bio")

    data = {
        "kalman": data_kalman,
        "emg_proc": data_emg,
        "markers": data_markers,
    }

    save(data, "data_pedalage_1.bio")

    data_loaded_2 = load("data_pedalage_1.bio")

    print(data_loaded)
