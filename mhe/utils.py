"""
This code provides some utility functions for the mhe implementation.
"""

import numpy as np
import bioptim
from biosiglive import save, load, OfflineProcessing, MskFunctions, InverseKinematicsMethods
from time import strftime
from scipy.interpolate import interp1d
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from bioptim import SolutionMerge, BiorbdModel
from casadi import MX, Function
import pandas as pd


def check_and_adjust_dim(*args):
    """
    Check if the dimensions of the arguments are the same.
    If not, the function will adjust the dimensions to be the same.
    """
    if len(args) == 1:
        conf = args[0]
    else:
        conf = {}
        for i in range(len(args)):
            for key in args[i].keys():
                conf[key] = args[i][key]
    return conf


def compute_force(
    sol: bioptim.Solution, get_force, nbmt: int, frame_to_save: int = 0, slide_size=1, save_all_frame=False
):
    """
    Compute the force.

    Parameters
    ----------
    sol: bioptim.Solution
        The solution of the mhe.
    get_force: function
        The function that computes the force.
    nbmt: int
        The number of muscles.
    use_excitation: bool
        If True, the excitation will be used.

    Returns
    -------
    Tuple of the force, joint angles, activation and excitation.
    """
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    if frame_to_save >= states["q"].shape[1] - 1 + slide_size:
        raise RuntimeError(
            f"You can ask to save frame from 0 to {states['q'].shape[1] + slide_size}." f"You asked {frame_to_save}."
        )
    slice_value = slice(frame_to_save, frame_to_save + slide_size)
    if not save_all_frame:
        q_est = states["q"][:, slice_value]
        dq_est = states["qdot"][:, slice_value]
        a_est = controls["muscles"][:, slice_value]
        f_ext = None if "f_ext" not in controls.keys() else controls["f_ext"][:, slice_value]
        tau_est = None if "tau" not in controls.keys() else controls["tau"][:, slice_value]
    else:
        q_est = states["q"]
        dq_est = states["qdot"]
        a_est = controls["muscles"]
        f_ext = None if "f_ext" not in controls.keys() else controls["f_ext"]
        tau_est = None if "tau" not in controls.keys() else controls["tau"]
    u_est = a_est
    force_est = np.zeros((nbmt, a_est.shape[1]))
    for i in range(nbmt):
        for j in range(a_est.shape[1]):
            force_est[i, j] = get_force(q_est[:, j], dq_est[:, j], a_est[:, j])[i, :]
    return q_est, dq_est, a_est, u_est, force_est, f_ext, tau_est


def map_activation(emg_proc, muscle_track_idx, model, emg_names, emg_init=None, mvc_normalized=True):
    act = np.zeros((len(muscle_track_idx), int(emg_proc.shape[1])))
    init_count = 0
    for j, name in enumerate(emg_names):
        count = 0
        for i in range(model.nbMuscles()):
            if name in model.muscleNames()[i].to_string():
                count += 1
        act[list(range(init_count, init_count + count)), :] = emg_proc[j, :]
        init_count += count
    return act


def interpolate_data(interp_factor: int, x_ref: np.ndarray, muscles_target: np.ndarray, markers_target: np.ndarray,
                     f_ext_target: np.ndarray
                     ):
    """
    Interpolate the reference and target data.

    Parameters
    ----------
    interp_factor: int
        The interpolation factor.
    x_ref: np.ndarray
        The reference x.
    muscles_target: np.ndarray
        The reference muscles.
    markers_target: np.ndarray
        The reference markers.

    Returns
    -------
    Tuple of interpolated data.
    """
    data = [x_ref, markers_target, muscles_target, f_ext_target]
    interp_data = []
    for d in data:
        interp_function = _interpolate_data_2d if len(d.shape) == 2 else _interpolate_data
        interp_data.append(interp_function(d, int(d.shape[-1] * interp_factor)))
    return interp_data[0], interp_data[1], interp_data[2], interp_data[3]


def _interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(markers_depth.shape[0]):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def _interpolate_data_2d(data, shape):
    new_data = np.zeros((data.shape[0], shape))
    x = np.linspace(0, 100, data.shape[1])
    f_mark = interp1d(x, data)
    x_new = np.linspace(0, 100, int(new_data.shape[1]))
    new_data = f_mark(x_new)
    return new_data


def _convert_string(string):
    return string.lower().replace("_", "")


def reorder_markers(markers, model, names):
    model_marker_names = [_convert_string(model.markerNames()[i].to_string()) for i in range(model.nbMarkers())]
    assert len(model_marker_names) == len(names)
    assert len(model_marker_names) == markers.shape[1]
    count = 0
    reordered_markers = np.zeros((markers.shape[0], len(model_marker_names), markers.shape[2]))
    for i in range(len(names)):
        if names[i] == "elb":
            names[i] = "elbow"
        if _convert_string(names[i]) in model_marker_names:
            reordered_markers[:, model_marker_names.index(_convert_string(names[i])),
            :] = markers[:, count, :]
            count += 1
    return reordered_markers


def force_func(biorbd_model: BiorbdModel):
    """
    Define the casadi function that compute the muscle force.

    Parameters
    ----------
    biorbd_model : BiorbdModel
        Model of the system.
    use_excitation : bool
        If True, use the excitation of the muscles.

    Returns
    -------
    Casadi function that compute the muscle force.
    """
    qMX = MX.sym("qMX", biorbd_model.nb_q, 1)
    dqMX = MX.sym("dqMX", biorbd_model.nb_q, 1)
    aMX = MX.sym("aMX", biorbd_model.nb_muscles, 1)
    return Function(
        "MuscleForce",
        [qMX, dqMX, aMX],
        [biorbd_model.muscle_forces(qMX, dqMX, aMX)],
        ["qMX", "dqMX", "aMX"],
        ["Force"],
    ).expand()


def get_data(ip=None, port=None, message=None, offline=False, offline_file_path=None):
    nfinal = -400
    n_init = 500
    if offline_file_path[-4:] == ".mat":
        mat = sio.loadmat(offline_file_path)
        x_ref, markers, muscles = mat["kalman"], mat["markers"], mat["emg_proc"]

    else:
        mat = load(offline_file_path)
        try:
            x_ref, markers, muscles = mat["kalman"][:, n_init:nfinal], mat["kin_target"][:, :, n_init:nfinal], mat["muscles_target"][:, n_init:nfinal]
        except:
            x_ref, markers, muscles = (
                mat["kalman"][:, n_init:nfinal],
                mat["markers"][:, :, n_init:nfinal],
                mat["emg_proc"][:, n_init:nfinal],
            )
    return x_ref, markers, muscles


def apply_params(model, file_path, with_casadi=True, ratio=True):
    data = load(file_path)
    param_list = data["p"]
    params_to_optim = data["optimized_params"]
    model = model.model if isinstance(model, BiorbdModel) else model
    for k in range(model.nbMuscles()):
        if "f_iso" in params_to_optim:
            f_init = model.muscle(k).characteristics().forceIsoMax()
            f_init = f_init.to_mx() if with_casadi else float(f_init)
            param_tmp = MX(param_list[params_to_optim.index("f_iso")][k]) if with_casadi else float(
                param_list[params_to_optim.index("f_iso")][k])
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setForceIsoMax(f_init * param_tmp)
        if "lm_optim" in params_to_optim:
            l_init = model.muscle(k).characteristics().optimalLength()
            l_init = l_init.to_mx() if with_casadi else float(l_init)
            param_tmp = MX(param_list[params_to_optim.index("lm_optim")][k]) if with_casadi else float(
                param_list[params_to_optim.index("lm_optim")][k])
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setOptimalLength(l_init * param_tmp)
            if ratio and "lt_slack" not in params_to_optim:
                lt_slack = model.muscle(k).characteristics().tendonSlackLength()
                lt_slack = lt_slack.to_mx() if with_casadi else float(lt_slack)
                ratio_tmp = lt_slack / l_init
                opt_l = model.muscle(k).characteristics().optimalLength().to_mx() if with_casadi else float(
                    model.muscle(k).characteristics().optimalLength())
                model.muscle(k).characteristics().setTendonSlackLength(opt_l * ratio_tmp)
        if "lt_slack" in params_to_optim:
            l_init = model.muscle(k).characteristics().tendonSlackLength()
            l_init = l_init.to_mx() if with_casadi else float(l_init)
            param_tmp = MX(param_list[params_to_optim.index("lt_slack")][k]) if with_casadi else float(
                param_list[params_to_optim.index("lt_slack")][k])
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setTendonSlackLength(l_init * param_tmp)
    return model


def load_mhe_results(data_path, n_frame_to_export):
    data = load(data_path, merge=False)
    key_to_export = ["q_est", "dq_est", "u_est", "muscle_force", "tau_est", "f_ext"]
    data_merged = {}
    for i in range(len(data)):
        for key in key_to_export:
            if data[i][key] is None:
                data_merged[key] = None
                continue
            data_merged[key] = data[i][key][..., n_frame_to_export:n_frame_to_export + 1] if i == 0 else np.append(
                data_merged[key], data[i][key][..., n_frame_to_export:n_frame_to_export + 1], axis=-1)
    return data_merged


def load_data(data_path, filter_depth, model, muscle_track_idx, emg_names, part,  source="depth", interp_factor=1, n_init=0, n_final=None):

    data = load(data_path)
    f_ext = data["f_ext"]
    names_from_source = data[source][f"marker_names"]
    markers = data[source][f"tracked_markers"]
    markers_tmp = markers.copy()
    markers[:, 6, :] = markers_tmp[:, 7, :]
    markers[:, 7, :] = markers_tmp[:, 6, :]

    #markers[:, 6, :], markers[:, 7, :] = markers[:, 7, :], markers[:, 6, :]

    emg = data["emg"]
    if isinstance(data["emg"], np.ndarray):
        muscles_target = map_activation(
            emg_proc=emg, muscle_track_idx=muscle_track_idx,
            model=model,
            emg_names=emg_names)
    else:
        muscles_target = np.zeros((model.nbMuscles(), markers.shape[2]))
    if model.nbQ() != data[source]["q_raw"].shape[0]:
        coef = 6
    else:
        coef = 0
    x_ref = np.concatenate((data[source]["q_raw"][coef:, :], data[source]["q_dot"][coef:, :]), axis=0)

    # import bioviz
    # b = bioviz.Viz(loaded_model=model)
    # b.load_movement(x_ref[:16, :])
    # b.load_experimental_markers(markers)
    # b.exec()
    n_final = n_final if n_final is not None else x_ref.shape[1]
    x_ref = x_ref[:, n_init:n_final]
    markers_target = markers[:, :, n_init:n_final]
    muscles_target = muscles_target[:, n_init:n_final]
    f_ext = f_ext[:, n_init:n_final]
    offline_data = [x_ref.copy(),
                    markers_target.copy(), muscles_target.copy(), f_ext.copy()]
    x_ref, markers_target, muscles_target, f_ext = interpolate_data(
        interp_factor, x_ref, muscles_target, markers_target, f_ext,
    )
    return offline_data, markers_target, names_from_source, f_ext, muscles_target, x_ref


def get_tracking_idx(model, emg_names):
    muscle_list = []
    for i in range(model.nbMuscles()):
        muscle_list.append(model.muscleNames()[i].to_string())
    muscle_track_idx = []
    for i in range(len(emg_names)):
        for j in range(len(muscle_list)):
            if emg_names[i] in muscle_list[j]:
                muscle_track_idx.append(j)
    return muscle_track_idx
