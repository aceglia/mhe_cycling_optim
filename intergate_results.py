import numpy as np
import matplotlib.pyplot as plt
from biosiglive import load
from mhe.utils import apply_params, load_mhe_results
import biorbd

def tau_from_muscles(model, q, qdot, act):
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(act[k])
    return model.muscularJointTorque(muscles_states, q, qdot).to_array()

def forward_dynamics(model, states, tau_tot, f_ext):
    q = states[:model.nbQ()]
    qdot = states[model.nbQ():]
    B = [0, 0, 0, 1]
    all_jcs = model.allGlobalJCS(q)
    RT = all_jcs[-1].to_array()
    # A = RT @ A
    B = RT @ B
    vecteur_OB = B[:3]
    f_ext[:3] = f_ext[:3] + np.cross(vecteur_OB, f_ext[3:6])
    ext = model.externalForceSet()
    ext.add("hand_left", f_ext)
    ddq = model.ForwardDynamics(q, qdot, tau_tot, ext).to_array()
    return np.concatenate((qdot, ddq), axis = 0)

def next_x(h, q, qdot, tau, f_ext, fun, model):
    states = np.concatenate((q, qdot), axis = 0)
    k1 = fun(model, states, tau, f_ext)
    k2 = fun(model, states + h / 2 * k1, tau, f_ext)
    k3 = fun(model, states + h / 2 * k2,  tau, f_ext)
    k4 = fun(model, states + h * k3, tau, f_ext)
    states = states + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    q = states[:model.nbQ()]
    qdot = states[model.nbQ():]
    return q, qdot

if __name__ == '__main__':
    part = "P10"
    trial = "gear_20"
    prefix = "/mnt/shared"
    model_dir = prefix + f"/Projet_hand_bike_markerless/RGBD/{part}/models"
    parameters_file_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/P10/result_optim_param.bio"

    with_param = False
    if with_param:
        model = f"{model_dir}/{trial}_model_scaled_dlc_ribs_new_seth_static_root_params.bioMod"
    #     biorbd_model = apply_params(biorbd_model, parameters_file_path, with_casadi=False,
    #                                            ratio=True)
    else:
        model = f"{model_dir}/{trial}_model_scaled_dlc_ribs_new_seth_static_root.bioMod"
    biorbd_model = biorbd.Model(model)
    result_dir = f"results/{part}"
    results = load_mhe_results(result_dir + f"/result_mhe_{trial}_dlc_optim_param_{with_param}.bio", 0)
    h = (1 / 240) / 4
    q, qdot = None, None
    all_states = np.zeros((results["q_est"].shape[0] * 2, results["q_est"].shape[1] - 70 + 1))
    for n in range(40, results["q_est"].shape[1]):
        if q is None:
            q = results["q_est"][:, n]
            qdot = results["dq_est"][:, n]
            all_states[:, n-40] = np.concatenate((q, qdot), axis=0)

        tau = results["tau_est"][:, n] + tau_from_muscles(biorbd_model, q, qdot, results["u_est"][:, n])
        f_ext = results["f_ext"][:, n]
        try:
            q, qdot = next_x(h, q, qdot, tau, f_ext, forward_dynamics, biorbd_model)
            all_states[:, n-40 + 1] = np.concatenate((q, qdot), axis = 0)
        except:
            print("Integration stoped at id ", n)
            break

    plt.figure("q")
    for i in range(biorbd_model.nbQ()):
        plt.subplot(3, 4, i + 1)
        plt.plot(results["q_est"][i, 40:n -1 ], 'r')
        plt.plot(all_states[i, :n - 40 -1], '--')

    plt.figure("qdot")
    for i in range(biorbd_model.nbQ()):
        plt.subplot(3, 4, i + 1)
        plt.plot(results["dq_est"][i, 40:n -1], 'r')
        plt.plot(all_states[i + biorbd_model.nbQ(), :n - 40 - 1], '--')
    plt.show()




