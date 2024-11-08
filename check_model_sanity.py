import biorbd
import biorbd_casadi as biorbd_ca
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from optim_params.file_io_utils import get_all_file
from biosiglive import load
from casadi import MX


def check_muscle_sanity(model_path, q, q_dot, plot_passive=True, plot_moment_arm=True, plot_length=True, color="b"):
    model = biorbd.Model(model_path)
    moment_arm = np.zeros((model.nbMuscles(), q.shape[0], q.shape[1]))
    length = np.zeros((model.nbMuscles(), q.shape[1]))
    velocity = np.zeros((model.nbMuscles(), q.shape[1]))

    length_ca = np.zeros((model.nbMuscles(), q.shape[1]))

    mus_passive = np.zeros((model.nbMuscles(), q.shape[1]))
    init_f_iso = MX.zeros(model.nbMuscles())
    init_l_opt = MX.zeros(model.nbMuscles())
    for m in range(model.nbMuscles()):
        init_f_iso[m] = model.muscle(m).characteristics().forceIsoMax()
        init_l_opt[m] = model.muscle(m).characteristics().optimalLength()

    def len_fct(model, q):
        # model.UpdateKinematicsCustom(q)
        # model.updateMuscles(q, True)
        mus_list = MX.zeros(model.nbMuscles(), 1)
        for i in range(model.nbMuscles()):
            mus_list[i] = model.muscle(i).length(model, q).to_mx()
        return mus_list

    def compute_tau_from_muscle(model, q, qdot, mus_act, p_f, p_lm):
        muscles_states = model.stateSet()
        for m in range(model.nbMuscles()):
            muscles_states[m].setActivation(mus_act[m])
            model.muscle(m).characteristics().setForceIsoMax(
                model.muscle(m).characteristics().forceIsoMax().to_mx() * (p_f[m]))
            model.muscle(m).characteristics().setOptimalLength(
                model.muscle(m).characteristics().optimalLength().to_mx() * (p_lm[m]))
        muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
        return model.muscularJointTorque(muscles_force, q, qdot).to_mx()

    model_ca = biorbd_ca.Model(model_path)
    q_sym = ca.MX.sym("q", model_ca.nbQ())
    qdot_sym = ca.MX.sym("qdot", model_ca.nbQdot())
    mus_act = ca.MX.sym("mus_act", model_ca.nbMuscles())
    p_f_sym = ca.MX.sym("p_f_sym", model_ca.nbMuscles())
    p_lm_sym = ca.MX.sym("p_lm_sym", model_ca.nbMuscles())
    mjt = compute_tau_from_muscle(model_ca, q_sym, qdot_sym, mus_act, p_f_sym, p_lm_sym)
    cas_fct = ca.Function("len_fct", [q_sym, qdot_sym, mus_act, p_f_sym, p_lm_sym], [mjt]).expand()
    # compute jacobian of cas_fct for each parameter using casadi
    jac_fct_f = ca.jacobian(mjt, p_f_sym)
    jac_fct_l = ca.jacobian(mjt, p_lm_sym)
    jac_fct_f = ca.Function("jac_fct_f", [q_sym, qdot_sym, mus_act, p_f_sym, p_lm_sym], [jac_fct_f]).expand()
    jac_fct_l = ca.Function("jac_fct_t", [q_sym, qdot_sym, mus_act, p_f_sym, p_lm_sym], [jac_fct_l]).expand()
    # compute jacobian of cas_fct for each parameter using numerical method
    jac_f_num = np.zeros((model_ca.nbGeneralizedTorque(), model_ca.nbMuscles(), q.shape[1]))
    jac_l_num = np.zeros((model_ca.nbGeneralizedTorque(), model_ca.nbMuscles(), q.shape[1]))
    for i in range(q.shape[1]):
        for m in range(model_ca.nbMuscles()):
            p_f_tmp = 1  #init_f_iso[m]
            p_lm_tmp = 1  # init_l_opt[m]
            jac_f_num[:, m:m + 1, i] = ca.Function("pouet", [MX()], [
                jac_fct_f(MX(q[:, i]), MX(q_dot[:, i]), MX.ones(model_ca.nbMuscles()) * 0.3, p_f_tmp, p_lm_tmp)])()[
                                           "o0"][:, m]
            jac_l_num[:, m:m + 1, i] = ca.Function("pouet_l", [MX()], [
                jac_fct_l(MX(q[:, i]), MX(q_dot[:, i]), MX.ones(model_ca.nbMuscles()) * 0.3, p_f_tmp, p_lm_tmp)])()[
                                           "o0"][:, m]

    [name.to_string() for name in model.nameDof()]
    colors = plt.cm.get_cmap("tab20", model.nbMuscles())
    plt.figure("Jac f_")
    for j in range(model.nbMuscles()):
        plt.subplot(6, 7, j + 1)
        for i in range(6, model.nbDof()):
            plt.plot(jac_f_num[i, j, :], label=model.nameDof()[i].to_string())
            plt.title(model.muscleNames()[j].to_string())
    plt.legend([name.to_string() for name in model.nameDof()][6:])
    plt.figure("Jac l_")
    for j in range(model.nbMuscles()):
        plt.subplot(6, 7, j + 1)
        for i in range(6, model.nbDof()):
            plt.plot(jac_l_num[i, j, :], label=model.nameDof()[i].to_string())
            plt.title(model.muscleNames()[j].to_string())
    plt.legend([name.to_string() for name in model.nameDof()][6:])

    #q = np.zeros_like(q)
    #q[4, :] = np.linspace(-5 *3.14/180, 17 *3.14/180, q.shape[1])
    # passive_torque = np.zeros((model.nbGeneralizedTorque(), q.shape[1]))
    # k1 = -50
    # k2 = -2
    # b1 = 1
    # b2 = 1
    # taueq = 0
    # wmax = 8
    # qmid = 0
    # deltap = -1
    # sv = 0.8
    # passive_torque_num = np.zeros((model.nbGeneralizedTorque(), q.shape[1]))
    # for i in range(q.shape[1]):
    #     passive_torque_num[:, i] = (b1 * np.exp(k1 * (q[:, i]-qmid)) + b2 * np.()(k2 * (q[:, i]-qmid))) * (1-deltap*0/(sv*wmax)) * (q[:, i] - deltap) + taueq
    #     passive_torque[:, i] = model.passiveJointTorque(q[:, i], np.zeros_like(q[:, i])).to_array()
    # plt.figure("passive_force")
    # # plt.plot(q[4, :], passive_torque[4, :])
    # plt.plot(q[4, :], passive_torque_num[4, :])
    #plt.show()
    #q[4, :] = np.linspace(-5, 17, q.shape[1])
    model_ca = biorbd_ca.Model(model_path)

    q_sym = ca.MX.sym("q", model_ca.nbQ())
    ml = len_fct(model_ca, q_sym)
    cas_fct = ca.Function("len_fct", [q_sym], [ml]).expand()
    for i in range(model.nbMuscles()):
        f_iso = model.muscle(i).characteristics().forceIsoMax()
        l_optim = model.muscle(i).characteristics().optimalLength()
        model.muscle(i).characteristics().setOptimalLength(l_optim)
        l_slack = model.muscle(i).characteristics().tendonSlackLength()
        model.muscle(i).characteristics().setTendonSlackLength(l_slack)

        print(f"ratio for muscle {model.muscleNames()[i].to_string()}", l_slack / l_optim)

    passive_torque = np.zeros((model.nbGeneralizedTorque(), q.shape[1]))
    # q = np.zeros_like(q)
    mus_fvce = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_flce = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_f_tot = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_torque = np.zeros((model.nbGeneralizedTorque(), model.nbMuscles(), q.shape[1]))
    for i in range(q.shape[1]):
        passive_torque[:, i] = model.passiveJointTorque(q[:, i], np.zeros_like(q[:, i])).to_array()
        moment_arm[:, :, i] = -model.musclesLengthJacobian(q[:, i]).to_array()
        muscle_states = model.stateSet()
        for m in range(model.nbMuscles()):
            muscle_states[m].setActivation(0.1)
        #mus_torque[:, :, i] = model.muscularJointTorque(muscle_states, q[:, i], q_dot[:, i]).to_array()
        # length_ca[:, i] = cas_fct(ca.MX(q[:, i]))
        length_ca[:, i] = ca.Function("pouet", [MX()], [cas_fct(ca.MX(q[:, i]))])()["o0"].toarray().squeeze()
        for m in range(model.nbMuscles()):
            mus_tmp = biorbd.HillDeGrooteType(model.muscle(m))
            #model.UpdateKinematicsCustom(q[:, i])
            #model.updateMuscles(q[:, i], True)
            length[m, i] = model.muscle(m).length(model, q[:, i])  #
            velocity[m, i] = model.muscle(m).velocity(model, q[:, i], q_dot[:, i])
            mus_tmp.length(model, q[:, i])
            mus_tmp.velocity(model, q[:, i], q_dot[:, i], True)
            mus_tmp.computeFlPE()
            mus_tmp.computeFlCE(muscle_states[m])
            mus_tmp.computeFvCE()
            mus_flce[m, i] = mus_tmp.FlCE(muscle_states[m]) * mus_tmp.characteristics().forceIsoMax()
            mus_fvce[m, i] = mus_tmp.FvCE() * mus_tmp.characteristics().forceIsoMax()
            mus_f_tot[m, i] = mus_tmp.characteristics().forceIsoMax() * (0.5 * mus_tmp.FlCE(muscle_states[m]) * mus_tmp.FvCE())
            mus_passive[m, i] = mus_tmp.FlPE() * mus_tmp.characteristics().forceIsoMax()
    # for i in range(model.nbMuscles()):
    #     max_ma = np.max(moment_arm[i, ...])
    #     max_moment_arm = np.where(moment_arm[i, ...] == max_ma)[0]
    #     print(f"max moment arm of {max_ma} for muscle {model.muscleNames()[i].to_string()} at index {model.nameDof()[int(max_moment_arm)].to_string()}")

    for m in range(model.nbMuscles()):
        norm_len = length[m, :] / model.muscle(m).characteristics().optimalLength()
        if 0.5 < norm_len.max() < 1.5 and 0.5 < norm_len.min() < 1.5:
            continue
        else:
            param = optimize_parameters_init(length[m, :], model.muscle(m).characteristics().optimalLength())
    if plot_passive:
        plt.figure("passive_force")
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            plt.plot(mus_passive[i, :], color)
            #plt.plot(mus_fvce[i, :], ".-", c=color)
            #plt.plot(mus_flce[i, :], ".-", c=color)
            plt.plot(np.repeat(model.muscle(i).characteristics().forceIsoMax(), q.shape[1]),"--", c=color)

            plt.title(model.muscleNames()[i].to_string())
    if plot_moment_arm:
        # for j in range(q.shape[0]):
        #     plt.figure("moment_arm_" + model.nameDof()[j].to_string())
        #     for i in range(model.nbMuscles()):
        #         plt.subplot(6, 7, i + 1)
        #         plt.plot(moment_arm[i, j, :])
        #         plt.title(model.muscleNames()[i].to_string())
        plt.figure("moment_arm_")
        for j in range(model.nbMuscles()):
            plt.subplot(6, 7, j + 1)
            for i in range(0, model.nbDof()):
                plt.plot(moment_arm[j, i, :], label=model.nameDof()[i].to_string())
                plt.title(model.muscleNames()[j].to_string())
        plt.legend([name.to_string() for name in model.nameDof()])
        plt.figure("torque")
        for j in range(model.nbMuscles()):
            plt.subplot(6, 7, j + 1)
            for i in range(6, model.nbDof()):
                plt.plot(moment_arm[j, i, :] * mus_passive[j, :], label=model.nameDof()[i].to_string())
                #plt.plot(moment_arm[j, i, :] * mus_f_tot[j, :], ".-", label=model.nameDof()[i].to_string())

                plt.title(model.muscleNames()[j].to_string())
            plt.gca().set_prop_cycle(None)
        plt.legend([name.to_string() for name in model.nameDof()][6:])
    if plot_length:
        plt.figure("velocity")
        max_vel = 5
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            plt.plot(velocity[i, :] / max_vel, color)
            #plt.plot(length_ca[i, :])
            plt.plot(np.repeat(max_vel/ max_vel, q.shape[1]), "--", c=color)
            plt.title(model.muscleNames()[i].to_string())
    if plot_length:
        plt.figure("norm_length")
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            if i==9:
                print(model.muscle(i).characteristics().optimalLength())
            plt.plot(length[i, :] / model.muscle(i).characteristics().optimalLength(), color)
            plt.title(model.muscleNames()[i].to_string())
        plt.figure("length")
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            if i == 9:
                print(model.muscle(i).characteristics().optimalLength())
            plt.plot(length[i, :], color)
            plt.title(model.muscleNames()[i].to_string())
            # plt.ylim([0, 1])
        # plt.ylim([0, 1])
    plt.show()


def optimize_parameters_init(all_length, param_value):
    from scipy.optimize import minimize
    # Define the function with a penalty if the parameter goes out of the range
    def objective(x, lengths, lower_bound, upper_bound):
        # If x is within bounds, return zero penalty (objective is zero)
        norm_len = (lengths / (param_value * x))
        if lower_bound <= norm_len.min() <= upper_bound and lower_bound <= norm_len.max() <= upper_bound:
            return 0
        # Apply a penalty proportional to the distance from the nearest bound if out of range
        elif lower_bound >= norm_len.min():
           return  (lower_bound - norm_len.min()) ** 2
        elif upper_bound <= norm_len.max():
            return (norm_len.max() - upper_bound) ** 2
        else:
            return 0


    # Set bounds and initial guess
    lower_bound = 0.5
    upper_bound = 1.5
    initial_guess = 1.0  # Initial guess outside the range

    # Run optimization to find the parameter value within bounds
    result = minimize(objective, initial_guess, args=(all_length, lower_bound, upper_bound), bounds=[(lower_bound, upper_bound)])

    # Extract optimized parameter
    optimized_parameter = result.x[0]
    return optimized_parameter


if __name__ == '__main__':
    participants = [f"P{i}" for i in range(10, 17)]
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/reference_data"
    model_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/"
    files, part = get_all_file(participants, data_dir, to_include=["reference_torque_gear_20"])
    data= load(files[0])
    end_idx = 1000
    q = data["q_ocp"][..., :end_idx]
    q_dot = data["q_dot_ocp"][..., :end_idx]
    model = model_dir + f"/P10/output_models/gear_20_model_scaled_dlc_ribs_new_seth_param.bioMod"
    check_muscle_sanity(model, q, q_dot, plot_passive=True, plot_moment_arm=True, plot_length=True, color="r")