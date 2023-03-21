"""
Script to scale the opensim model then translate it into biomod file and initialize it with a Kalman filter.
Data can be live-streamed or prerecorded to avoid the subject waiting.
"""
import os

try:
    import biorbd
    import bioviz
except ModuleNotFoundError:
    pass
import numpy as np
from biosiglive import load, save
import matplotlib.pyplot as plt
from pathlib import Path

def compute_qdot(q):
    dt = 0.01
    qdot = (q[:, 1:] - q[:, :q.shape[1] - 1]) / dt
    qdot = np.concatenate((qdot, qdot[:, qdot.shape[1] - 1:qdot.shape[1]]), axis=1)
    return qdot

def kalman_func(markers, model):
    markersOverFrames = []
    freq = 100  # Hz
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)
    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)
    for i in range(markers.shape[2]):
        markersOverFrames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])
    q_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    q_dot = np.ndarray((model.nbQ(), len(markersOverFrames)))
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        q_dot[:, i] = Qdot.to_array()
    return q_recons, q_dot

def initialize(
    biomod_model,
        file_path,
        markers,
        save_data=False):
    """
    Initialize the model with a Kalman filter and/or scale it.
    Parameters
    ----------
    model_path : str
        Path of the model to initialize
    data_dir : str
        Path of the directory where the data are stored
    scaling : bool, optional
        If True, the model will be scaled using opensim. The default is False.
    off_line : bool, optional
        If True, the model will be initialized and scaled with prerecorded data. The default is True.
    mass : int, optional
        Mass of the subject. The default is None.
    """

    model_path = biomod_model
    bmodel = biorbd.Model(model_path)
    # q_recons, q_dot = kalman_func(markers[:, :, :], model=bmodel)
    ik = biorbd.InverseKinematics(bmodel, markers)
    ik.solve("trf")
    q_recons = ik.q
    q_mean = q_recons.mean(axis=1)
    print(q_mean[3], q_mean[4], q_mean[5], " xyz ", q_mean[0], q_mean[1], q_mean[2])
    q_dot = compute_qdot(q_recons)
    if save_data:
        data = load(file_path)
        data["kalman"] = np.concatenate((q_recons, q_dot), axis=0)
        if os.path.isfile(file_path):
            os.remove(file_path)
        save(data, file_path)

    b = bioviz.Viz(model_path=model_path, show_muscles=False, mesh_opacity=1, show_local_ref_frame=False)
    b.load_movement(q_recons)  # Q from kalman array(nq, nframes)
    b.load_experimental_markers(markers)  # experimental markers array(3, nmarkers, nframes)
    b.exec()
    plt.show()


if __name__ == "__main__":
    file_path = "/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/data_gen/saves/pedalage_1_proc.bio"
    data_dir = "/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/results"
    bio_model = f"{data_dir}/wu_gauche_cycling_pos_scaled_1.bioMod"
    markers = load(file_path)["markers"]
    initialize(biomod_model=bio_model, markers=markers, file_path=file_path, save_data=True)
