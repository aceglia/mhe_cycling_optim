import matplotlib.pyplot as plt
import numpy as np

def plot_muscle_activation(muscle_activations, emg=None, muscle_names=None, muscle_track_idx=None, q=None):
    """
    Plots the muscle activations over time.
    Parameters
    ----------
    """
    plt.figure(f"Muscle activations")
    if muscle_names is None:
        muscle_names = ['Muscle'+ str(i) for i in range(len(muscle_activations))]
    for i, act in enumerate(muscle_activations):
        plt.subplot(int(np.ceil(muscle_activations.shape[0]/4)), 4, i+1)
        plt.plot(act, color="b")
        if emg is not None and i in muscle_track_idx:
            plt.plot(emg[muscle_track_idx.index(i), :], color="r")
        plt.plot(q[-2, :] / 10, color="g")
        plt.title(muscle_names[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Muscle activation')

def plot_muscle_force(model, muscle_activations, q, q_dot, muscle_names=None):
    """
    Plots the muscle activations over time.
    Parameters
    ----------
    """
    import biorbd
    mus_fvce = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_flce = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_flpe = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_fce = np.zeros((model.nbMuscles(), q.shape[1]))
    for i in range(q.shape[1]):
        muscle_states = model.stateSet()
        for m in range(model.nbMuscles()):
            model.UpdateKinematicsCustom(q[:, i])
            # model.updateMuscles(q[:, i], True)
            mus_tmp = biorbd.HillDeGrooteType(model.muscle(m))
            muscle_states[m].setActivation(muscle_activations[m, i])
            # mus_tmp.characteristics().setMaxShorteningSpeed(500)
            # mus_tmp = model.muscle(m)
            mus_tmp.length(model, q[:, i])
            mus_tmp.velocity(model, q[:, i], q_dot[:, i], True)
            mus_tmp.computeFlPE()
            mus_tmp.computeFlCE(muscle_states[m])
            mus_tmp.computeFvCE()
            mus_flce[m, i] = mus_tmp.FlCE(muscle_states[m]) * mus_tmp.characteristics().forceIsoMax()
            mus_fce[m, i] = (muscle_activations[m, i] * mus_tmp.FlCE(muscle_states[m]) * mus_tmp.FvCE()) * mus_tmp.characteristics().forceIsoMax()
            mus_fvce[m, i] = mus_tmp.FvCE() * mus_tmp.characteristics().forceIsoMax()
            mus_flpe[m, i] = mus_tmp.FlPE() * mus_tmp.characteristics().forceIsoMax()

    plt.figure(f"Muscle forces")
    if muscle_names is None:
        muscle_names = ['Muscle' + str(i) for i in range(len(muscle_activations))]
    for m in range(len(muscle_names)):
        plt.subplot(int(np.ceil(muscle_activations.shape[0] / 4)), 4, m + 1)
        # plt.plot(mus_flce[m, :], color="b")
        plt.plot(mus_fce[m, :], color="g")
        plt.plot(mus_flpe[m, :], color="r")
        plt.title(muscle_names[m])
        plt.xlabel('Time (s)')
        plt.ylabel('Muscle activation')

def plot_joint_torques(muscle_torques, reference_torques, residuals_torques=None, muscle_torque_optim=None, joint_names=None):
    plt.figure(f"Joint torques")
    if residuals_torques is None:
        residuals_torques = np.zeros_like(muscle_torques)
    if joint_names is None:
        joint_names = ['Joint'+ str(i) for i in range(len(muscle_torques))]
    for i in range(0, reference_torques.shape[0]):
        plt.subplot(int(np.ceil(reference_torques.shape[0]/4)), 4, i + 1)
        plt.plot(reference_torques[i, :], color="r")
        plt.plot(muscle_torques[i, :] + residuals_torques[i, :], color="b")
        plt.plot(residuals_torques[i, :], color="g", alpha=0.5)
        if muscle_torque_optim is not None:
            plt.plot(muscle_torque_optim[i, :] , ".-", color="y")
        plt.title(joint_names[i])

def plot_param(param, muscle_names=None, param_name=None, bounds=None, initial_value=1):
    if muscle_names is None:
        muscle_names = ['Muscle'+ str(i) for i in range(param[1].shape[0])]
    if param_name is None:
        param_name = [f"Parameter {i}" for i in range(len(param))]
    for p in range(len(param)):
        plt.figure(param_name[p])
        bar_width = 0.1
        indices = np.linspace(0, len(muscle_names), len(muscle_names))
        # Separate data into above and below 1
        above_one = np.clip(np.array(param[p]) - 1, 0, None)  # Values above 1
        below_one = np.clip(np.array(param[p]) - 1, None, 0)  # Values below 1
        # Plot above 1 values
        plt.bar(indices, above_one, width=bar_width)
        plt.bar(indices, below_one, width=bar_width)
        plt.xticks(indices + bar_width / 2, muscle_names, rotation=90)
        y_ticks = np.arange(bounds[p][0] - 1, bounds[p][1] - 1, 0.1)  # Define range for y-ticks
        y_labels = [f'{1 + tick:.1f}' for tick in y_ticks]  # Create labels centered on 1
        plt.yticks(y_ticks, y_labels)
        # Center y-axis at 1
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=bounds[p][1] - 1, color='black', linestyle='--')
        plt.axhline(y=bounds[p][0] - 1, color='black', linestyle='--')
        # Labels and title
        plt.ylabel('Deviation from initial value')

def plot_norm_length(model, q):
    length = np.zeros((model.nbMuscles(), q.shape[1]))
    for i in range(q.shape[1]):
        for m in range(model.nbMuscles()):
            model.UpdateKinematicsCustom(q[:, i])
            model.updateMuscles(q[:, i], True)
            length[m, i] = model.muscle(m).length(model, q[:, i])
    plt.figure("norm_length")
    for i in range(model.nbMuscles()):
        plt.subplot(6, 7, i + 1)
        plt.plot(length[i, :] / model.muscle(i).characteristics().optimalLength())
        plt.title(model.muscleNames()[i].to_string())
