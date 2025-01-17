import os
import time

from optim_params.parameters_identifier import ParametersIdentifier
from optim_params.identification_utils import get_all_muscle_len
from optim_params.enum import Parameters
from optim_params.file_io_utils import get_all_file, get_data_dict
import biorbd
from biosiglive import load
import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    Solver,
    InterpolationType,
    SolutionMerge,
    MovingHorizonEstimator,
    DynamicsFunctions,
    DynamicsEvaluation,
    NonLinearProgram,
    Node,
    ConfigureProblem,
)
from matplotlib import pyplot as plt
from casadi import MX, vertcat


def custom_torque_driven(
    time: MX,
    states: MX,
    controls: MX,
    parameters: MX,
    stochastic_variables: MX,
    nlp: NonLinearProgram,
):
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    from casadi import cross

    f_ext = DynamicsFunctions.get(nlp.controls["f_ext"], controls)
    B = [0, 0, 0, 1]
    all_jcs = nlp.model.model.allGlobalJCS(q)
    RT = all_jcs[-1].to_mx()
    B = RT @ B
    vecteur_OB = B[:3]
    f_ext[:3] = f_ext[:3] + cross(vecteur_OB, f_ext[3:6])
    ext = nlp.model.model.externalForceSet()
    ext.add("hand_left", f_ext)
    ddq = nlp.model.model.ForwardDynamics(q, qdot, tau, ext).to_mx()
    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):  # , with_f_ext: bool):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_new_variable(
        "f_ext", ["mx", "my", "mz", "fx", "fy", "fz"], ocp, nlp, as_states=False, as_controls=True
    )
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_torque_driven)


def _compute_new_bounds(data, with_floating_base=False):
    bounds = [
        # "rotations xyz // thorax\n\t\ttranslations xyz // thorax\n\t\t//ranges \n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n",
        "rotations xyz // thorax\n\t\ttranslations xyz // thorax\n",  # thorax
        "rotations x\n\t\tranges\n\t\t\t\t-3 3",  # clavicle
        "rotations y\n\t\tranges\n\t\t\t\t-3 3",  # clavicle
        "//rotations z\n\t\t//ranges\n\t\t\t\t//-3 3",  # clavicle
        # "rotations xyz\n\t\tranges\n\t\t\t\t-0.2 1",  # scapula
        "rotations xyz\n\t\tranges\n\t\t\t\t-3 3\n\t\t\t\t-3 3\n\t\t\t\t-3 3",  # scapula
        "rotations x\n\t\tranges\n\t\t\t\t-3 3",  # shoulder
        "rotations y\n\t\tranges\n\t\t\t\t-3 3",  # shoulder
        "rotations z\n\t\tranges\n\t\t\t\t-3 3",  # shoulder
        "rotations z\n\t\tranges\n\t\t\t\t-0.5 3",  # elbow
        "rotations y\n\t\tranges\n\t\t\t\t-3 3",  # forearm
        "//rotations xy\n\t\t//ranges\n\t\t\t\t//0.1 0.8",  # Hand
    ]
    if with_floating_base is False:
        # data = data.replace("31.4054400268435", "1 //31.4054400268435")
        bounds[0] = "//rotations xyz // thorax\n\t\t//translations xyz // thorax\n"
    # bounds = [
    #     "rotations xyz // thorax\n\t\ttranslations xyz // thorax\n\t\tranges \n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n\t\t-0.5 0.5\n",
    #     "rotations x\n\t\tranges\n\t\t\t\t-0.7 0.2",  # clavicle
    #     "rotations y\n\t\tranges\n\t\t\t\t-0.5 0.5",  # clavicle
    #     "//rotations z\n\t\tranges\n\t\t\t\t-3 3",  # clavicle
    #     # "rotations xyz\n\t\tranges\n\t\t\t\t-0.2 1",  # scapula
    #     "rotations xyz\n\t\tranges\n\t\t\t\t-0.1 1\n\t\t\t\t-0.1 0.8\n\t\t\t\t-0.2 0.5",  # scapula
    #     "rotations x\n\t\tranges\n\t\t\t\t-0.4 0.8",  # shoulder
    #     "rotations y\n\t\tranges\n\t\t\t\t0.2 1",  # shoulder
    #     "rotations z\n\t\tranges\n\t\t\t\t-1.2 0.2",  # shoulder
    #     "rotations z\n\t\tranges\n\t\t\t\t0.8 2.2",  # elbow
    #     "rotations y\n\t\tranges\n\t\t\t\t0.3 0.8",  # forearm
    # ]
    data_tmp = data
    idx_start = 0
    count = 0
    while True:
        if count == 0:
            idx_start = data_tmp.find("rotations xyz // thorax")
            to_replace = "rotations xyz // thorax\n\t\ttranslations xyz // thorax"
            idx_start = idx_start + len(to_replace)
            data_tmp = data_tmp.replace(to_replace, bounds[0])
            count += 1
        else:
            break
            idx_start = data_tmp.find("rotations", idx_start)
            if idx_start == -1:
                break
            idx_end = data_tmp.find("endsegment", idx_start) - 2
            data_tmp = data_tmp.replace(data_tmp[idx_start:idx_end], bounds[count])
            idx_start = idx_end
            count += 1
    return data_tmp


def prepare_ocp(
    biorbd_model_path, q, qdot, n_shooting, tau, final_time, with_floating_base, downsampling_factor, f_ext
):
    first_idx = 0 if with_floating_base else 6
    bio_model = BiorbdModel(biorbd_model_path)
    q = q[:, ::downsampling_factor]
    qdot = qdot[:, ::downsampling_factor]
    tau = tau[:, ::downsampling_factor]
    q = q[first_idx:, : n_shooting + 1]
    qdot = qdot[first_idx:, : n_shooting + 1]
    # qdot[2, :] = 0 * np.ones_like(qdot[2, :])
    tau = tau[first_idx:, :n_shooting]
    f_ext = f_ext[:, ::downsampling_factor][:, :n_shooting]
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=100000000, target=q, node=Node.ALL)
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", weight=1000000000, target=qdot, node=Node.ALL
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_CONTROL,
        key="f_ext",
        weight=100000000,
        target=f_ext,
        node=Node.ALL_SHOOTING,
        multi_thread=False,
    )

    dynamics = DynamicsList()
    # dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(custom_configure, dynamic_function=custom_torque_driven, expand_dynamics=True)
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    # x_bounds.add("q", min_bound=q, max_bound=q, interpolation=InterpolationType.EACH_FRAME)
    # x_bounds.add("qdot", min_bound=qdot, max_bound=qdot, interpolation=InterpolationType.EACH_FRAME)
    # x_bounds.add("qdot", min_bound=[-10000] * bio_model.nb_tau, max_bound=[1000] * bio_model.nb_tau)

    x_init = InitialGuessList()
    x_init.add("q", q, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", qdot, interpolation=InterpolationType.EACH_FRAME)
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[-1000] * bio_model.nb_tau, max_bound=[1000] * bio_model.nb_tau)

    u_init = InitialGuessList()
    u_init.add("tau", tau, interpolation=InterpolationType.EACH_FRAME)
    u_bounds.add("f_ext", min_bound=[-12000] * 6, max_bound=[12000] * 6)
    u_init.add("f_ext", f_ext, interpolation=InterpolationType.EACH_FRAME)
    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=False,
        n_threads=8,
    )


if __name__ == "__main__":
    with_param = True
    with_residual_torque = False
    use_ratio_tracking = True
    save_data = False
    participants = [f"P{i}" for i in range(10, 11)]
    params_to_optimize = [
        Parameters.f_iso,
        Parameters.lm_optim,
    ]  # , Parameters.lt_slack] #]# [Parameters.lm_optim, , Parameters.lt_slack] #
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/reference_data"
    model_dir = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/"
    files, part = get_all_file(
        participants, data_dir, to_include=["reference_torque_gear_20_with_technical_marker_params.bio"]
    )
    date = time.strftime("%Y-%m-%d_%H-%M")
    for file, participant in zip(files, part):
        list_tmp = file.replace(".bio", "").split("/")[-1].split("_")
        trial_short = "gear_" + list_tmp[list_tmp.index("gear") + 1]
        model_path = (
            model_dir
            + f"/{participant}/output_models/{trial_short}_model_scaled_dlc_technical_marker_params_tmp.bioMod"
        )
        suffix = "_ocp"
        ocp_result = load(file)
        q = ocp_result["q" + suffix]
        qdot = ocp_result["q_dot" + suffix]
        tau = ocp_result["tau" + suffix]
        f_ext = ocp_result["f_ext" + suffix]
        for i in range(2):
            with_floating_base = i == 1

            with open(model_path, "r") as f:
                data = f.read()
            data = _compute_new_bounds(data, with_floating_base=with_floating_base)
            new_biorbd_model_path = model_path.replace("_tmp.bioMod", "_test.bioMod")
            with open(new_biorbd_model_path, "w") as f:
                f.write(data)
            ocp = prepare_ocp(
                new_biorbd_model_path,
                q=q,
                qdot=qdot,
                n_shooting=60,
                tau=tau,
                final_time=1,
                with_floating_base=with_floating_base,
                downsampling_factor=2,
                f_ext=f_ext,
            )
            solver = Solver.IPOPT()
            solver.set_linear_solver("ma57")
            solver.set_convergence_tolerance(1e-6)
            solver.set_maximum_iterations(100)
            sol = ocp.solve(solver=solver)
            merged_states = sol.decision_states(to_merge=SolutionMerge.NODES)
            merged_controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
            colors = ["b", "g"]
            to_plot = ["q", "qdot", "tau", "f_ext"]
            to_plot_ref = ["q_ocp", "q_dot_ocp", "tau_ocp", "f_ext_ocp"]
            for j in range(len(to_plot)):
                plt.figure(to_plot[j])
                count = 0
                range_idx = 16 if to_plot[j] != "f_ext" else 6
                for k in range(range_idx):
                    plt.subplot(range_idx // 4 + 1, 4, k + 1)
                    if k < 6 and i == 0:
                        continue
                    else:
                        idx_to_plot = count if i == 0 else k
                        if j < 2:
                            plt.plot(
                                merged_states[to_plot[j]][idx_to_plot, :],
                                label=str(with_floating_base),
                                color=colors[i],
                            )
                        else:
                            plt.plot(
                                merged_controls[to_plot[j]][idx_to_plot, :],
                                label=str(with_floating_base),
                                color=colors[i],
                            )
                        count += 1
                    if i == 1:
                        plt.plot(ocp_result[to_plot_ref[j]][:, ::2][k, :61], label="ref", color="r")
    plt.legend()
    plt.show()
