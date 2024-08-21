"""
TODO: Cleaning
This is a basic example on how to use muscle driven to perform an optimal reaching task.
The arm must reach a marker while minimizing the muscles activity and the states. The problem is solved using both
ACADOS and Ipopt.
"""
import os

import matplotlib.pyplot as plt
from casadi import MX, horzcat, vertcat, sum1
import numpy as np
import biorbd_casadi as biorbd_ca
import casadi as ca
import bioptim
import biorbd
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
    ParameterList,
    MovingHorizonEstimator,
    DynamicsFunctions,
    DynamicsEvaluation,
    NonLinearProgram,
    Node,
    ConfigureProblem,
    BiMappingList,
)


def compute_tau_from_muscle(model, q, qdot, muscles_act, params):
    muscles_tau = model.muscularJointTorque(muscles_act * params, q, qdot).to_mx()
    return muscles_tau


def muscles_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
):
    """
    Forward dynamics driven by muscle.

    Parameters
    ----------
    states: MX.sym
        The state of the system
    controls: MX.sym
        The controls of the system
    parameters: MX.sym
        The parameters of the system
    nlp: NonLinearProgram
        The definition of the system
    with_contact: bool
        If the dynamic with contact should be used
    fatigue: FatigueDynamicsList
        To define fatigue elements
    with_torque: bool
        If the dynamic should be added with residual torques
    """

    DynamicsFunctions.apply_parameters(parameters, nlp)
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    mus_act_nlp, mus_act = (nlp.states, states) if "muscles" in nlp.states else (nlp.controls, controls)
    mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles"], mus_act)
    muscles_tau = compute_tau_from_muscle(nlp.model.model, q, qdot, mus_activations, nlp.parameters.mx)

    tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, False)

    dxdt = MX(nlp.states.shape, ddq.shape[1])
    dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
    dxdt[nlp.states["qdot"].index, :] = ddq

    has_excitation = True if "muscles" in nlp.states else False
    if has_excitation:
        mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
        dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations)
        dxdt[nlp.states["muscles"].index, :] = horzcat(*[dmus for _ in range(ddq.shape[1])])

    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def custom_torque_driven(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        stochastic_variables: MX,
        nlp: NonLinearProgram,
        # with_f_ext: bool = False
):
    # DynamicsFunctions.apply_parameters(parameters, nlp)
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    # tau[:6] = MX(0)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    # if with_f_ext:

    # f_ext = DynamicsFunctions.get(nlp.states["f_ext"], states)
    # # ddq = nlp.model.forward_dynamics(q, qdot, tau, f_ext)
    # A = [0, 0, 0, 1]
    # B = [0, 0, 0, 1]
    # ind_1 = 25
    # ind_2 = 40
    # force_locale = change_ref_between_two_segments(ind_1, ind_2, model, f_ext[:, 0])
    # all_jcs = model.all_global_jcs(q)
    # RT = all_jcs[ind_1].to_mx()
    # RT2 = all_jcs[ind_2].to_mx()
    # B = RT @ B
    # A = RT2 @ A
    # vecteur_BA = A[:3] - B[:3]
    # force_global[:3] = f_ext[:3, 0] + cross(vecteur_BA, f_ext[3:6, 0])
    # force_global[3:] = f_ext[3:, 0]
    from casadi import cross
    f_ext = DynamicsFunctions.get(nlp.controls["f_ext"], controls)
    B = [0, 0, 0, 1]
    all_jcs = nlp.model.model.allGlobalJCS(q)
    RT = all_jcs[-1].to_mx()
    # A = RT @ A
    B = RT @ B
    vecteur_OB = B[:3]
    f_ext[:3] = f_ext[:3] + cross(vecteur_OB, f_ext[3:6])
    # force_global = change_ref_for_global(ind_1, q, model, force_locale)
    #ddq = nlp.model.forward_dynamics(q, qdot, tau, force_global)
    ext = nlp.model.model.externalForceSet()
    ext.add("hand_left", f_ext)
    ddq = nlp.model.model.ForwardDynamics(q, qdot, tau, ext).to_mx()
    # else:
    # ddq = nlp.model.forward_dynamics(q, qdot, tau, MX.zeros(6, 1))
    # ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, False)

    # dxdt = MX(nlp.states.shape, ddq.shape[1])
    # dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
    # dxdt[nlp.states["qdot"].index, :] = ddq
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
    # if with_f_ext:
    # ConfigureProblem.configure_f_ext(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_new_variable("f_ext", ["mx", "my", "mz", "fx", "fy", "fz"],
                                            ocp, nlp, as_states=False, as_controls=True)
    # ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_torque_driven)  # , with_f_ext=with_f_ext)

def custom_func_min_len(controller) -> MX:
    #all_len = len_fct(controller.model.model, controller.q.cx)
    len = controller.mx_to_cx("len", controller.model.muscle_length, controller.q)
    return len

def modify_isometric_force(biorbd_model, value, fiso_init):
    for k in range(biorbd_model.nb_muscles):
        biorbd_model.muscle(k).characteristics().setForceIsoMax(value[k] * fiso_init[k])


def prepare_ocp(biorbd_model_path, final_time, n_shooting,
                # params,
                x_warm=None, use_sx=False, n_threads=1,
                mhe=False, param_bounds=None, target=None, i=None,
                with_muscle=False,
                q_init=None,
                param=None,
                pas_tau=None,
                with_f_ext=False,
                f_ext=None,
                track_previous=False):
    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max, tau_init = -10000, 10000, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.2
    # tau_mapping = BiMappingList()
    # to_second = list(range(bio_model.nb_q - 6))
    # to_second = [None] * 6 + to_second
    # to_first = list(range(6, bio_model.nb_q))
    # tau_mapping.add("tau", to_second=to_second, to_first=to_first)
    # for m in range(bio_model.nb_muscles):
    #     bio_model.muscle(m).setForceIsoMax(params[m])

    # Add objective functions
    objective_functions = ObjectiveList()

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10000, multi_thread=False,
                            index=list(range(abs(10 - bio_model.nb_q), bio_model.nb_q)))

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, multi_thread=False,
                            index=list(range(abs(10 - bio_model.nb_q), bio_model.nb_q)))
    if with_f_ext:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="f_ext", weight=1000000000,
                                target=f_ext[:, :n_shooting], node=Node.ALL_SHOOTING, multi_thread=False)

    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=1000000000,
                            target=target[:, :3, :n_shooting + 1],
                            node=Node.ALL, multi_thread=False,
                            marker_index=list(range(3)))
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=1000000000,
                            target=target[:, 3:, :n_shooting + 1],
                            node=Node.ALL, multi_thread=False,
                            marker_index=list(range(3, bio_model.nb_markers)))
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, multi_thread=False,
    #                         index=list(range(6))
    #                         )
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=10000000, target=q_init[: bio_model.nb_q, :n_shooting+1],
    #                         node=Node.ALL, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=10, multi_thread=False,
                            index=list(range(abs(10 - bio_model.nb_q), bio_model.nb_q)))
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=100, multi_thread=False,
    #                         index=[3])
    if track_previous:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=10,
                                target=q_init[: bio_model.nb_q, :n_shooting + 1],
                                node=Node.ALL, multi_thread=False)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", weight=1000,
                                target=q_init[bio_model.nb_q: bio_model.nb_q * 2, :n_shooting + 1],
                                node=Node.ALL, multi_thread=False)
    if with_muscle:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=100000,
                                multi_thread=False)

    if bio_model.nb_q > 10:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, multi_thread=False,
                                index=list(range(abs(10 - bio_model.nb_q))))
        # if bio_model.nb_tau != q_init.shape[0] - bio_model.nb_q:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=10, multi_thread=False,
                                index=list(range(abs(10 - bio_model.nb_q))))
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1000, multi_thread=False,
                                index=list(range(abs(10 - bio_model.nb_q))))

    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1000000, multi_thread=False,
    #                         derivative=True,
    #                         index=list(range(abs(10 - bio_model.nb_q), bio_model.nb_q)))
    # objective_functions.add(
    #     custom_func_min_len,
    #     custom_type=ObjectiveFcn.Lagrange,
    #     node=Node.ALL,
    #     quadratic=True,
    #     weight=100000000000,
    # )

    def compute_tau_from_muscle(nlp, q, qdot, mus_act, fatigue):
        model = nlp.model.model
        muscles_states = model.stateSet()
        for m in range(model.nbMuscles()):
            muscles_states[m].setActivation(mus_act[m])
        muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx() * param
        return model.muscularJointTorque(muscles_force, q, qdot).to_mx() + pas_tau

    dynamics = DynamicsList()
    if with_muscle:
        bioptim.dynamics.dynamics_functions.DynamicsFunctions.compute_tau_from_muscle = compute_tau_from_muscle
        dynamics.add(DynamicsFcn.MUSCLE_DRIVEN)
    else:
        if with_f_ext:
            dynamics.add(custom_configure,
                         dynamic_function=custom_torque_driven,
                         # with_f_ext=with_f_ext,
                         expand_dynamics=True)
        else:
            # dynamics.add(custom_configure,
            #              dynamic_function=custom_torque_driven,
            #              # with_f_ext=with_f_ext,
            #              expand_dynamics=True)
            dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", min_bound=[-1000] * bio_model.nb_tau, max_bound=[1000] * bio_model.nb_tau)
    # x_bounds[:, 0] = (1.0, 1.0, 0, 0)
    x_init = InitialGuessList()
    # Initial guess
    x_init.add("q", q_init[: bio_model.nb_q, :n_shooting + 1], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", q_init[bio_model.nb_q:, :n_shooting + 1], interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    u_bounds = BoundsList()

    u_init = InitialGuessList()

    if with_muscle:
        u_bounds.add(
            [muscle_min] * bio_model.nb_muscles,
            [muscle_max] * bio_model.nb_muscles,
        )
        u_init.add([muscle_init] * bio_model.nb_muscles)
    else:
        u_bounds.add("tau", min_bound=[tau_min] * bio_model.nb_tau, max_bound=[tau_max] * bio_model.nb_tau)
        u_init.add("tau", [tau_init] * bio_model.nb_tau)
        if with_f_ext:
            u_bounds.add("f_ext", min_bound=[-12000] * 6, max_bound=[12000] * 6)
            u_init.add("f_ext", f_ext[:, :n_shooting], interpolation=InterpolationType.EACH_FRAME)

    # Get initial isometric forces
    # f_iso = []
    # for k in range(bio_model.nb_muscles):
    #     f_iso.append(bio_model.muscle(k).characteristics().forceIsoMax().to_mx())
    # f_iso = [798.52, 624.3, 435.56, 624.3, 624.3, 987.26]
    # Define the parameter to optimize
    # param_min = param_max = param_init = params
    # if param_bounds is None:
    #     bound_p_iso = Bounds(
    #         [0.2] * bio_model.nb_muscles, [6] * bio_model.nb_muscles, interpolation=InterpolationType.CONSTANT)
    #     p_iso_init = InitialGuess([1.0] * bio_model.nb_muscles)
    # else:
    # bound_p_iso = Bounds(
    #     param_min, param_max, interpolation=InterpolationType.CONSTANT)
    # # param_init = (param_bounds[1] + param_bounds[0]) / 2
    # p_iso_init = InitialGuess(param_init)

    # if use_parameters:
    # parameters = ParameterList()
    # parameters.add(
    #     "p_iso",  # The name of the parameter
    #     modify_isometric_force,  # The function that modifies the biorbd model
    #     p_iso_init,
    #     bound_p_iso,  # The bounds
    #     size=bio_model.nb_muscles,  # The number of elements this particular parameter vector has
    #     fiso_init=f_iso,
    # )
    # ------------- #
    if not mhe:
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
            use_sx=use_sx,
            n_threads=n_threads,
            # variable_mappings=tau_mapping,
            # parameters=parameters,
        )
    else:
        return MovingHorizonEstimator(
            bio_model=bio_model,
            dynamics=dynamics,
            window_len=n_shooting,
            window_duration=final_time,
            common_objective_functions=objective_functions,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            n_threads=n_threads,
            use_sx=use_sx,
            # variable_mappings=tau_mapping,
            # parameters=parameters,
        )


def get_solver_options(solver):
    mhe_dict = {"solver_first_iter": None, "solver": solver}
    solver_options = {
        # "sim_method_jac_reuse": 1,
        # "levenberg_marquardt": 50.0,
        # "nlp_solver_step_length": 0.9,
        "qp_solver_iter_max": 1000,
    }

    if isinstance(solver, Solver.ACADOS):
        # mhe_dict["solver"].set_maximum_iterations(10)
        for key in solver_options.keys():
            mhe_dict["solver"].set_option_unsafe(val=solver_options[key], name=key)
        mhe_dict["solver"].set_convergence_tolerance(1e-4)
        from copy import copy
        mhe_dict["solver"] = copy(mhe_dict["solver"])
        mhe_dict["solver"].set_print_level(1)
        mhe_dict["solver"].set_qp_solver("PARTIAL_CONDENSING_HPIPM")  # PARTIAL_CONDENSING_OSQP PARTIAL_CONDENSING_HPIPM
        mhe_dict["solver"].set_integrator_type("IRK")
        mhe_dict["solver"].set_maximum_iterations(100)
        mhe_dict["solver"].set_convergence_tolerance(1e-5)

    elif isinstance(solver, Solver.IPOPT):
        mhe_dict["solver"].set_hessian_approximation("exact")
        # mhe_dict["solver"].set_limited_memory_max_history(50)
        mhe_dict["solver"].set_maximum_iterations(50)
        mhe_dict["solver"].set_print_level(5)
        mhe_dict["solver"].set_tol(1e-5)
        mhe_dict["solver"].set_linear_solver("ma57")
        from copy import copy
        mhe_dict["solver_first_iter"] = copy(mhe_dict["solver"])
        mhe_dict["solver_first_iter"].set_maximum_iterations(100)
        mhe_dict["solver_first_iter"].set_tol(1e-5)

    else:
        raise NotImplementedError("Solver not recognized")
    return mhe_dict


def _map_activation(emg_proc, muscle_track_idx, model, emg_names, emg_init=None, mvc_normalized=True):
    if mvc_normalized:
        emg_proc = emg_init
    act = np.zeros((len(muscle_track_idx), int(emg_proc.shape[1])))
    act_init = np.zeros((len(muscle_track_idx), int(emg_proc.shape[1])))
    init_count = 0
    for j, name in enumerate(emg_names):
        count = 0
        for i in range(model.nbMuscles()):
            if name in model.muscleNames()[i].to_string():
                count += 1
        act[list(range(init_count, init_count + count)), :] = emg_proc[j, :]
        if emg_init is not None:
            act_init[list(range(init_count, init_count + count)), :] = emg_init[j, :]
        init_count += count
    return act, act_init


def _return_param_from_mapping(p_mapping, p_init, scaling_factor=1):
    final_param_list = []
    count = 0
    for mapping in p_mapping:
        p_tmp = None
        for m in mapping[1]:
            if p_tmp is None:
                p_tmp = p_init[mapping[0].index(m) + count] / scaling_factor
            else:
                p_tmp = ca.vertcat(p_tmp, p_init[mapping[0].index(m) + count] / scaling_factor)
        final_param_list.append(p_tmp)
        count += len(mapping[0])
    return final_param_list


def _apply_params(model, param_list, params_to_optim, model_param_init=None, with_casadi=True, ratio=None):
    for k in range(model.nbMuscles()):
        if "f_iso" in params_to_optim:
            f_init = model.muscle(k).characteristics().forceIsoMax() if not model_param_init else \
            model_param_init[params_to_optim.index("f_iso")][k]
            # param_tmp = param_list[params_to_optim.index("f_iso")][k] * f_init
            # param_tmp = param_tmp if with_casadi else float(param_tmp)
            # model.muscle(k).characteristics().setForceIsoMax(f_init + param_tmp)
            param_tmp = param_list[params_to_optim.index("f_iso")][k]
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setForceIsoMax(f_init * param_tmp)
        if "lm_optim" in params_to_optim:
            l_init = model.muscle(k).characteristics().optimalLength() if not model_param_init else \
            model_param_init[params_to_optim.index("lm_optim")][k]
            # param_tmp = param_list[params_to_optim.index("lm_optim")][k] * l_init
            # param_tmp = param_tmp if with_casadi else float(param_tmp)
            # model.muscle(k).characteristics().setOptimalLength(l_init + param_tmp)
            param_tmp = param_list[params_to_optim.index("lm_optim")][k]
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setOptimalLength(l_init * param_tmp)
            if ratio and "lt_slack" not in params_to_optim:
                opt_l = model.muscle(k).characteristics().optimalLength().to_mx() if with_casadi else float(
                    model.muscle(k).characteristics().optimalLength())
                model.muscle(k).characteristics().setTendonSlackLength(opt_l * ratio[k])
        if "lt_slack" in params_to_optim:
            l_init = model.muscle(k).characteristics().tendonSlackLength() if not model_param_init else \
            model_param_init[params_to_optim.index("lt_slack")][k]
            # param_tmp = param_list[params_to_optim.index("lt_slack")][k] * l_init
            # param_tmp = param_tmp if with_casadi else float(param_tmp)
            # model.muscle(k).characteristics().setTendonSlackLength(l_init + param_tmp)
            param_tmp = param_list[params_to_optim.index("lt_slack")][k]
            param_tmp = param_tmp if with_casadi else float(param_tmp)
            model.muscle(k).characteristics().setTendonSlackLength(l_init * param_tmp)
    return model


def _muscle_torque(model, scaling_factor,
                   p_mapping,
                   muscle_list, use_p_mapping=True,
                   with_param=True,
                   return_casadi_function=False, params_to_optim=None, model_params_init=None,
                   ratio=None):

    def muscle_joint_torque(model, activations, q, qdot,
                            p_init=None
                            ) -> MX:
        if with_param:
            param_list = _return_param_from_mapping(p_mapping, p_init)
            model = _apply_params(model, param_list, params_to_optim, model_params_init, ratio=ratio)
        muscles_states = model.stateSet()
        for k in range(model.nbMuscles()):
            muscles_states[k].setActivation(activations[k])
        # muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx() * p_init[:model.nbMuscles()]
        # mjt = model.muscularJointTorque(muscles_force, q, qdot).to_mx()
        mjt = model.muscularJointTorque(muscles_states, q, qdot).to_mx()
        return mjt

    if return_casadi_function:
        if with_param:
            x = ca.MX.sym("x", model.nbMuscles())
            q = ca.MX.sym("q", model.nbQ())
            qdot = ca.MX.sym("qdot", model.nbQ())
            n_p = len(p_mapping) * model.nbMuscles()
            p_sym = ca.MX.sym("p_sym_bis", n_p)
            mjt = muscle_joint_torque(model, x, q, qdot,
                                      p_sym)

            mjt_func = ca.Function("mjt_func", [x, q, qdot,
                                                p_sym
                                                ], [mjt]).expand()
        # else:
        #     mjt = muscle_joint_torque(model, x, q, qdot,)
        #     mjt_func = ca.Function("mjt_func", [x, q, qdot], [mjt]).expand()
            return mjt_func

    mus_j_torque = ca.MX.zeros((q.shape[0], q.shape[1]))
    for k in range(q.shape[1]):
        act = x[k * model.nbMuscles(): (k + 1) * model.nbMuscles()] / scaling_factor[0]
        mus_j_torque[:, k] = muscle_joint_torque(model, act, q[:, k], qdot[:, k], p_init)
    return mus_j_torque, p_init


def _add_to_J(J, weigth, value):
    J += (weigth * value) ** 2
    return J


def _return_forward_function(model):

    def forward_dynamics(states_sym, tau_tot_sym, f_ext_sym):
        from casadi import cross
        B = [0, 0, 0, 1]
        all_jcs = model.allGlobalJCS(q_sym)
        RT = all_jcs[-1].to_mx()
        # A = RT @ A
        B = RT @ B
        vecteur_OB = B[:3]
        f_ext = ca.MX(f_ext_sym)
        f_ext[:3] = f_ext[:3] + cross(vecteur_OB, f_ext[3:6])
        ext = model.externalForceSet()
        ext.add("hand_left", f_ext)
        q = states_sym[:model.nbQ()]
        qdot = states_sym[model.nbQ():]
        ddq = model.ForwardDynamics(q, qdot, tau_tot_sym, ext).to_mx()
        return ca.vertcat(qdot, ddq)
        #return model.ForwardDynamics(q_sym, qdot_sym, tau_tot_sym, ext).to_mx()

    q_sym = ca.MX.sym("q", model.nbQ())
    qdot_sym = ca.MX.sym("qdot", model.nbQdot())
    tau_tot_sym = ca.MX.sym("tau_tot", model.nbQ())
    state_sym = ca.vertcat(q_sym, qdot_sym)
    f_ext_sym = ca.MX.sym("f_ext", 6)
    forward = forward_dynamics(state_sym, tau_tot_sym, f_ext_sym)
    forward_function = ca.Function("forward", [state_sym, tau_tot_sym, f_ext_sym], [forward]).expand()

    def next_x(h, q, qdot, tau, f_ext, fun):
        k1 = fun(vertcat(q, qdot), tau, f_ext)
        k2 = fun(vertcat(q, qdot) + h / 2 * k1, tau, f_ext)
        k3 = fun(vertcat(q, qdot) + h / 2 * k2,  tau, f_ext)
        k4 = fun(vertcat(q, qdot) + h * k3, tau, f_ext)
        states = vertcat(q, qdot) + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q = states[:model.nbQ()]
        qdot = states[model.nbQ():]
        return q, qdot

    return forward_function, next_x


def _compute_cost_function_old(model, scaling_factor, x, q, qdot, tau, p, pas_tau, tau_mat, act, weights,
                           p_mapping=None,
                           mus_j_torque=None,
                           f_ext=None,
                           mvc_normalized=True,
                           torque_as_constraint=False,
                           with_torque=True,
                           dynamics_as_constraint=False,
                           muscle_track_idx=None,
                           muscle_casadi_function=None,
                           passive_joint_torques=None,
                           with_param=True,
                           params_to_optim=None,
                           use_ratio_tracking=False,
                           ratio=None,
                           param_init=None,
                           passive_torque_idx=None,
                           forward_function=None,
                           next_x=None,
                           h=None,
                           ):
    ns = q.shape[1]
    with_root = True if model.nbQ() > 10 else False
    penalty = []
    J = 0
    if with_torque:
        J = _add_to_J(J, weights["min_pas_torque"], pas_tau)
        # J = _add_to_J(J, weights["min_pas_torque"], pas_tau[2:])
        # J = _add_to_J(J, weights["min_pas_torque"] / 1000, pas_tau[1])
    lm_opti, lt_slack = None, None
    if with_param:
        count = 0
        for p_idx in range(len(p_mapping)):
            # J = _add_to_J(J, weights[f"min_{params_to_optim[p_idx]}"], (p[count: count + len(p_mapping[p_idx][0])]))
            J = _add_to_J(J, weights[f"min_{params_to_optim[p_idx]}"],
                          (p[count: count + len(p_mapping[p_idx][0])] - 1 * scaling_factor[1]))
            # if use_ratio_tracking:
            #     if params_to_optim[p_idx] == "lm_optim":
            #         lm_opti = p[count: count + len(p_mapping[p_idx][0])]
            #     if params_to_optim[p_idx] == "lt_slack":
            #         lt_slack = p[count: count + len(p_mapping[p_idx][0])]
            count += len(p_mapping[p_idx][0])

        if use_ratio_tracking and "lt_slack" in params_to_optim and "lm_optim" in params_to_optim:
            to_minimize = None
            for i in range(lm_opti.shape[0]):
                lm_init = param_init[params_to_optim.index("lm_optim")][i]
                lt_init = param_init[params_to_optim.index("lt_slack")][i]
                # lm_optimized = lm_init + lm_opti[i] * lm_init
                # lt_slack_optimized = lt_init + lt_slack[i] * lt_init
                lm_optimized = lm_init * lm_opti[i]
                lt_slack_optimized = lt_init * lt_slack[i]
                #to_minimize = (lt_slack_optimized / lm_optimized) - ratio[i] if to_minimize is None else ca.vertcat(to_minimize, (lt_slack_optimized / lm_optimized)-ratio[i])
                J = _add_to_J(J, weights["ratio_tracking"], (lt_slack_optimized / lm_optimized) - ratio[i])

            J = _add_to_J(J, weights["ratio_tracking"], to_minimize)

    if not mvc_normalized:
        all_x = MX.zeros(model.nbMuscles(), ns)
        max_list = []
        for m in range(model.nbMuscles()):
            for k in range(ns):
                all_x[m, k] = x[k * model.nbMuscles() + m]
            max_list.append(ca.mmax(all_x[m, :]))

    tau_init = None if not with_torque else np.zeros((len(passive_torque_idx) * ns))
    constr = None
    for i in range(ns):
        if muscle_casadi_function is not None:
            if with_param:

                mus_tau = muscle_casadi_function(
                    x[i * model.nbMuscles(): (i + 1) * model.nbMuscles()] / scaling_factor[0],
                    q[:, i], qdot[:, i], p / scaling_factor[1])
            else:
                mus_tau = muscle_casadi_function(
                    x[i * model.nbMuscles(): (i + 1) * model.nbMuscles()] / scaling_factor[0],
                    q[:, i], qdot[:, i])
        else:
            mus_tau = mus_j_torque[:, i]
        if not torque_as_constraint:
            if with_torque:
                pas_tau_tmp = pas_tau[i * len(passive_torque_idx): (i + 1) * len(passive_torque_idx)]
                count = 0
                # final_pas_tau_tmp = ca.MX.zeros(model.nbQ() - 10, 1)
                final_pas_tau_tmp = None
                for t in range(0, model.nbGeneralizedTorque()):
                    if t in passive_torque_idx:
                        final_pas_tau_tmp = pas_tau_tmp[count] if final_pas_tau_tmp is None else ca.vertcat(
                            final_pas_tau_tmp, pas_tau_tmp[count])
                        count += 1
                    else:
                        final_pas_tau_tmp = ca.MX.zeros(1) if final_pas_tau_tmp is None else ca.vertcat(
                            final_pas_tau_tmp, ca.MX.zeros(1))
                tau_init[i * len(passive_torque_idx):(i + 1) * len(passive_torque_idx)] = np.zeros(
                    (len(passive_torque_idx)))
                # J = _add_to_J(J, weights["tau_tracking"],
                #               (tau[first_idx:, i] * scaling_factor[2] + passive_joint_torques[first_idx:, i] - (mus_tau[first_idx:] * scaling_factor[2] + final_pas_tau_tmp)))
                J = _add_to_J(J, weights["tau_tracking"],
                              (tau[passive_torque_idx, i] * scaling_factor[2] + passive_joint_torques[
                                  passive_torque_idx, i] - (
                                           mus_tau[passive_torque_idx] * scaling_factor[2] + final_pas_tau_tmp[
                                       passive_torque_idx])))
                # J = _add_to_J(J, weights["tau_tracking"] / 10, (tau[passive_torque_idx, i] - mus_tau[passive_torque_idx]))
            else:
                J = _add_to_J(J, weights["tau_tracking"],
                              (tau[model.nbGeneralizedTorque() - 10:, i] - mus_tau[model.nbGeneralizedTorque() - 10:]))
        else:
            if constr is None:
                constr = (tau[:, i] - (mus_tau + pas_tau[:] / scaling_factor[2])) ** 2
            else:
                constr = vertcat(constr, ((tau[:, i] - (mus_tau + pas_tau[:] / scaling_factor[2])) ** 2))

        # if i < ns - 1:
        #     next_q, next_q_dot = next_x(h, q[:, i], qdot[:, i], (mus_tau + pas_tau / scaling_factor[2]),
        #                                 f_ext[:, i],
        #                                 forward_function)
        #     if dynamics_as_constraint:
        #         if constr is None:
        #             constr = (next_q - q[:, i + 1])
        #             # constr = vertcat(constr, (next_q_dot - qdot[:, i + 1]))
        #         else:
        #             constr = vertcat(constr, (next_q - q[:, i + 1]))
        #             # constr = vertcat(constr, (next_q_dot - qdot[:, i + 1]))
        #     else:
        #         J = vertcat(J, weights["dynamics"] * ((next_q - q[:, i + 1]) ** 2))
    #     #         # J = vertcat(J, weights["dynamics"] * ((next_q_dot - qdot[:, i + 1]) ** 2))
    x0 = np.zeros((model.nbMuscles() * (ns), 1)) + 0.1 * scaling_factor[0]
    all_x_mus = None
    for i in range(ns):
        for m in range(model.nbMuscles()):
            if m in muscle_track_idx:
                idx = muscle_track_idx.index(m)
                x0[i * model.nbMuscles() + m] = act[idx, i] * scaling_factor[0]
                # J = _add_to_J(J, weights["activation_tracking"], ((x[
                #                                                    i * model.nbMuscles() + m: i * model.nbMuscles() + m + 1]
                #                                                    ) - MX(
                #act[idx, i] * scaling_factor[0])))
                if all_x_mus is None:
                    all_x_mus = weights["activation_tracking"] * ((x[
                                                                   i * model.nbMuscles() + m: i * model.nbMuscles() + m + 1]
                                                                  ) - MX(
                        act[idx, i] * scaling_factor[0]))
                else:
                    all_x_mus = vertcat(all_x_mus, weights["activation_tracking"] * ((x[
                                                                                      i * model.nbMuscles() + m: i * model.nbMuscles() + m + 1]
                                                                                     ) - MX(
                        act[idx, i] * scaling_factor[0])))

            else:
                x0[i * model.nbMuscles() + m] = 0.1 * scaling_factor[0]
                if all_x_mus is None:
                    all_x_mus = weights["min_act"] * x[i * model.nbMuscles() + m]
                else:
                    all_x_mus = vertcat(all_x_mus, weights["min_act"] * x[i * model.nbMuscles() + m])
                # J = _add_to_J(J, weights["min_act"], x[i * model.nbMuscles() + m])
    J = _add_to_J(J, 1, all_x_mus)
    # J += (weights["activation_tracking"] * sum1(((x))) )** 2
    x0 = np.zeros((model.nbMuscles() * (ns), 1)) + 0.1 * scaling_factor[0]
    constr = None
    tau_init = None if not with_torque else np.zeros((len(passive_torque_idx) * ns))
    return J, constr, x0, tau_init


def _add_params_to_J(p, p_mapping, params_to_optim, scaling_factor, use_ratio_tracking, weights, param_init=None,
                     ratio=None, bounds_l_norm=True, muscle_len=None):
    lm_opti, lt_slack = None, None
    count = 0
    J_params = 0
    g = []
    # p = ca.MX.sym("p_sym_bis", len(p_mapping) * len(p_mapping[0][0]))
    for p_idx in range(len(p_mapping)):
        # J = _add_to_J(J, weights[f"min_{params_to_optim[p_idx]}"],
        #               (p[count: count + len(p_mapping[p_idx][0])] - 1 * scaling_factor[1]))
        p_tmp = p[count: count + len(p_mapping[p_idx][0])]
        for p_idx_bis in range(p_tmp.shape[0]):
            J_params += (weights[f"min_{params_to_optim[p_idx]}"] * (p_tmp[p_idx_bis] - 1 * scaling_factor[1][p_idx])) ** 2
        if use_ratio_tracking:
            if params_to_optim[p_idx] == "lm_optim":
                lm_opti = p[count: count + len(p_mapping[p_idx][0])]
            if params_to_optim[p_idx] == "lt_slack":
                lt_slack = p[count: count + len(p_mapping[p_idx][0])]
        count += len(p_mapping[p_idx][0])
    if bounds_l_norm and "lm_optim" in params_to_optim and muscle_len is not None:
        ratio_l = ca.fabs(MX(muscle_len) / (lm_opti / MX(scaling_factor[1][params_to_optim.index("lm_optim")])))
        for i in range(ratio_l.shape[1]):
            g = vertcat(g, ratio_l[:, i])
        # for m in range(ratio_l.shape[0]):
        #     for k in range(ratio_l.shape[1]):
        #         if g is None:
        #             g = ratio_l[m, k]
        #         else:
        #             g += ratio_l[m, k]

    if use_ratio_tracking and "lt_slack" in params_to_optim and "lm_optim" in params_to_optim:
        for i in range(lm_opti.shape[0]):
            lm_init = param_init[params_to_optim.index("lm_optim")][i]
            lt_init = param_init[params_to_optim.index("lt_slack")][i]
            lm_optimized = lm_init * lm_opti[i]
            lt_slack_optimized = lt_init * lt_slack[i]
            ratio = lt_init / lm_init
            J_params += (weights["ratio_tracking"] * (lt_slack_optimized / lm_optimized - ratio)) ** 2
    #     J = _add_to_J(J, weights["ratio_tracking"], to_minimize)
    if isinstance(g, list) and len(g) == 0:
        g = None
    return J_params, g


def _add_x_to_J(model, x, act, weights, scaling_factor, muscle_track_idx, ns):
    all_x_mus = None
    J = 0
    for i in range(ns):
        for m in range(model.nbMuscles()):
            if m in muscle_track_idx:
                idx = muscle_track_idx.index(m)
                J += (weights["activation_tracking"] * ((x[i * model.nbMuscles() + m]) - MX(
                        act[idx, i] * scaling_factor[0]))) ** 2
            else:
                J += (weights["min_act"] * x[i * model.nbMuscles() + m]) ** 2
    return J


def _add_tau_to_J(model, tau, x, p, q, qdot, mus_j_torque, pas_tau, weights, scaling_factor, passive_torque_idx,
                  with_torque, ns, muscle_casadi_function, with_param, tau_as_constraint=False, p_mapping=None):
    J = 0
    mus_tau_tot = MX.zeros(model.nbGeneralizedTorque(), ns)
    for i in range(ns):
        if muscle_casadi_function is not None:
            if with_param:
                p_tmp = None
                count = 0
                for p_idx in range(len(p_mapping)):
                    n_p = len(p_mapping[p_idx][0])
                    p_tmp = p[count:count + n_p] / scaling_factor[1][p_idx] if p_tmp is None else vertcat(p_tmp, p[count:count + n_p] / scaling_factor[1][p_idx])
                    count += n_p
                mus_tau = muscle_casadi_function(
                    x[i * model.nbMuscles(): (i + 1) * model.nbMuscles()] / scaling_factor[0],
                    q[:, i], qdot[:, i]
                    , p_tmp
                )
            else:
                mus_tau = muscle_casadi_function(
                    x[i * model.nbMuscles(): (i + 1) * model.nbMuscles()] / scaling_factor[0],
                    q[:, i], qdot[:, i])
        else:
            mus_tau = mus_j_torque[:, i]

        mus_tau_tot[:, i] = mus_tau
        pas_tau_tmp = pas_tau[i * len(passive_torque_idx): (i + 1) * len(passive_torque_idx)] if with_torque else None
        count = 0
        for t in passive_torque_idx:
            to_substract = mus_tau[t] * scaling_factor[2] + pas_tau_tmp[count] if with_torque else mus_tau[t]
            sqrt = 1 if tau_as_constraint else 2
            #J = (tau[t, i] * scaling_factor[2] - to_substract) if J is None else vertcat(J, (tau[t, i] * scaling_factor[2] - to_substract))
            J += (weights["tau_tracking"] * (tau[t, i] * scaling_factor[2] - to_substract)) ** sqrt
            count += 1
    #J = (weights["tau_tracking"] * sum1(J)) ** sqrt
    return J, mus_tau_tot


def _add_dyn_to_J(h, q, qdot, mus_tau, pas_tau, f_ext, forward_function, ns, weights, scaling_factor, tau, next_x, state_int, passive_torque_idx):
    J=0
    non_muscle_tau = MX.zeros(q.shape[0], ns)
    for i in range(ns):
        pas_tau_tmp = pas_tau[i * len(passive_torque_idx): (i + 1) * len(passive_torque_idx)] / scaling_factor[2]
        count = 0
        for j in range(non_muscle_tau.shape[0]):
            if j < q.shape[0] - 10:
                non_muscle_tau[j, i] = tau[j, i]
            else:
                non_muscle_tau[j, i] = tau[j, i] + pas_tau_tmp[count, 0]
                count += 1

        tau_tot = mus_tau[:, i] + non_muscle_tau[:, i]
        next_q, next_q_dot = next_x(h, q[:, i], qdot[:, i], tau_tot,
                                        f_ext[:, i],
                                        forward_function)
        J += (weights["dynamics"] * (next_q - state_int[:q.shape[0], i])) ** 2
    return J


def _get_initial_values(model, passive_torque_idx, ns, muscle_track_idx, act, scaling_factor):
    tau_init = np.zeros((len(passive_torque_idx) * ns))
    x0 = np.zeros((model.nbMuscles() * ns, 1)) + 0.2 * scaling_factor[0]
    for i in range(ns):
        for m in range(model.nbMuscles()):
            if m in muscle_track_idx:
                idx = muscle_track_idx.index(m)
                x0[i * model.nbMuscles() + m] = act[idx, i] * scaling_factor[0]

    return x0, tau_init


def _get_all_muscle_len(model, q):
    mus_list = np.zeros((model.nbMuscles(), q.shape[1]))
    for i in range(q.shape[1]):
        for m in range(model.nbMuscles()):
            mus_list[m, i] = model.muscle(m).length(model, q[:, i])
            #print(mus_list / (model.muscle(m).characteristics().optimalLength()))
    return mus_list


def _get_cost_n_dependant(model, scaling_factor, x, q, qdot, tau, p, pas_tau, act, weights,
                           p_mapping=None,
                           with_torque=True,
                           muscle_track_idx=None,
                           muscle_casadi_function=None,
                           with_param=True,
                           passive_torque_idx=None,
                           tau_as_constraint=False):
    J = 0
    # min pas torque
    if with_torque:
        for tau_idx in range(pas_tau.shape[0]):
            J += (weights["min_pas_torque"] * pas_tau[tau_idx]) ** 2

    # min act
    for m in range(model.nbMuscles()):
        if m in muscle_track_idx:
            idx = muscle_track_idx.index(m)
            J += (weights["activation_tracking"] * ((x[m]) - MX(
                act[idx] * scaling_factor[0]))) ** 2
        else:
            J += (weights["min_act"] * x[m]) ** 2

    # track tau
    if with_param:
        p_tmp = None
        count = 0
        for p_idx in range(len(p_mapping)):
            n_p = len(p_mapping[p_idx][0])
            p_tmp = p[count:count + n_p] / scaling_factor[1][p_idx] if p_tmp is None else vertcat(p_tmp, p[count:count + n_p] / scaling_factor[1][p_idx])
            count += n_p
        mus_tau = muscle_casadi_function(x / scaling_factor[0], q, qdot, p_tmp)
    else:
        mus_tau = muscle_casadi_function(x / scaling_factor[0], q, qdot)
    pas_tau_tmp = pas_tau if with_torque else None
    count = 0
    for t in passive_torque_idx:
        to_substract = mus_tau[t] * scaling_factor[2] + pas_tau_tmp[count] if with_torque else mus_tau[t]
        sqrt = 1 if tau_as_constraint else 2
        factor = scaling_factor[2] if with_torque else 1
        J += (weights["tau_tracking"] * (tau[t] * factor - to_substract)) ** sqrt
        count += 1
    return J


def _get_cost_non_n_dependant(model, scaling_factor, x, q, qdot, tau, p, pas_tau, act, weights,
                          p_mapping=None,
                          with_torque=True,
                          muscle_track_idx=None,
                          muscle_casadi_function=None,
                          with_param=True,
                          passive_torque_idx=None,
                          tau_as_constraint=False):
    J = 0
    # min pas torque
    if with_torque:
        for tau_idx in range(pas_tau.shape[0]):
            J += (weights["min_pas_torque"] * pas_tau[tau_idx]) ** 2

    # min act
    for m in range(model.nbMuscles()):
        if m in muscle_track_idx:
            idx = muscle_track_idx.index(m)
            J += (weights["activation_tracking"] * ((x[m]) - MX(
                act[idx] * scaling_factor[0]))) ** 2
        else:
            J += (weights["min_act"] * x[m]) ** 2

    # track tau
    if with_param:
        p_tmp = None
        count = 0
        for p_idx in range(len(p_mapping)):
            n_p = len(p_mapping[p_idx][0])
            p_tmp = p[count:count + n_p] / scaling_factor[1][p_idx] if p_tmp is None else vertcat(p_tmp,
                                                                                                  p[count:count + n_p] /
                                                                                                  scaling_factor[1][
                                                                                                      p_idx])
            count += n_p
        mus_tau = muscle_casadi_function(x / scaling_factor[0], q, qdot, p_tmp)
    else:
        mus_tau = muscle_casadi_function(x / scaling_factor[0], q, qdot)
    pas_tau_tmp = pas_tau if with_torque else None
    count = 0
    for t in passive_torque_idx:
        to_substract = mus_tau[t] * scaling_factor[2] + pas_tau_tmp[count] if with_torque else mus_tau[t]
        sqrt = 1 if tau_as_constraint else 2
        J += (weights["tau_tracking"] * (tau[t] * scaling_factor[2] - to_substract)) ** sqrt
        count += 1
    return J


def _compute_cost_function(model, scaling_factor, x, q, qdot, tau, p, pas_tau, act, weights,
                           param_init,
                           p_mapping=None,
                           mus_j_torque=None,
                           with_torque=True,
                           muscle_track_idx=None,
                           muscle_casadi_function=None,
                           with_param=True,
                           params_to_optim=None,
                           use_ratio_tracking=False,
                           passive_torque_idx=None,
                           tau_as_constraint=False,
                           dyn_as_constraint=False,
                            forward_function=None,
                                    next_x=None,
                                    h=None,
                           f_ext = None,
                           state_int=None,
                           all_muscle_len=None,
                           l_norm_bounded=False
                           ):
    ns = q.shape[1]
    J = 0
    # min pas torque
    penalties = []
    g = []
    if with_torque:
        J_tau_tmp = 0
        for tau_idx in range(pas_tau.shape[0]):
            J_tau_tmp += (weights["min_pas_torque"] * pas_tau[tau_idx]) ** 2
        penalties.append(J_tau_tmp)
    # min params
    if with_param:
        # all_muscle_len = _get_all_muscle_len(model, q)
        J_params, g_params = _add_params_to_J(p, p_mapping, params_to_optim, scaling_factor, use_ratio_tracking, weights, param_init,
                                    bounds_l_norm=l_norm_bounded, muscle_len=all_muscle_len)
        penalties.append(J_params)
        if l_norm_bounded:
            g.append(g_params)
    # track and min act
    J_x = _add_x_to_J(model, x, act, weights, scaling_factor, muscle_track_idx, ns)
    penalties.append(J_x)
    #
    # track exp tau
    weights["tau_tracking"] = 1 if tau_as_constraint else weights["tau_tracking"]
    J_tau, mus_tau = _add_tau_to_J(model, tau, x, p, q, qdot, mus_j_torque, pas_tau, weights, scaling_factor,
                                    passive_torque_idx, with_torque, ns, muscle_casadi_function, with_param, p_mapping=p_mapping)

    if not tau_as_constraint:
        penalties.append(J_tau)
    else:
        g.append(J_tau)

    # weights["dynamics"] = 1 if dyn_as_constraint else weights["dynamics"]
    # J_dyn = _add_dyn_to_J(h, q, qdot, mus_tau, pas_tau, f_ext, forward_function, ns, weights, scaling_factor, tau, next_x, state_int, passive_torque_idx)
    # if not dyn_as_constraint:
    #     penalties.append(J_dyn)
    # else:
    #     g.append(J_dyn)

    J = penalties[0]
    for J_tmp in penalties[1:]:
        J = vertcat(J, J_tmp)
    J = sum1(J)

    if len(g) > 0:
        G = g[0]
        for g_tmp in g[1:]:
            G = vertcat(G, g_tmp)
    else:
        G = None
    return J, G


def _return_bounds(model, scaling_factor, p, ns, tau, x, pas_tau, x0, tau_0, with_param=True, with_torque=True,
                   p_init=1, params_to_optim=(), p_mapping=None, param_bounds=None, l_norm_bounded=False, torque_as_constr=False, dyn_as_constr=False):
    lbx = ca.DM.zeros(model.nbMuscles() * (ns)) + (0.0001) * scaling_factor[0]
    ubx = ca.DM.ones(model.nbMuscles() * (ns)) * scaling_factor[0]
    #x0 = ca.DM.ones(model.nbMuscles() * (ns)) * 0.2 * scaling_factor[0]
    lbg, ubg = None, None
    if l_norm_bounded:
        if "lm_optim" in params_to_optim:
            ubg = 1.6
            lbg = 0.4

    # lbg = ca.DM.zeros(tau.shape[0] * ns)
    # ubg = ca.DM.zeros(tau.shape[0] * ns)
    if with_param:
        lb_p = None
        ub_p = None
        init_p = None
        for p_idx, param in enumerate(params_to_optim):
            lb, ub = param_bounds[p_idx][0], param_bounds[p_idx][1]
            n_p = len(p_mapping[p_idx][0])
            init_p = ca.DM.zeros(n_p) + p_init * scaling_factor[1][p_idx] if init_p is None else ca.vertcat(init_p,
                                                                                             ca.DM.zeros(n_p) + p_init * scaling_factor[1][p_idx])
            lb_p = ca.DM.zeros(n_p) + lb * scaling_factor[1][p_idx] if lb_p is None else ca.vertcat(lb_p,
                                                                                             ca.DM.zeros(n_p) + lb *
                                                                                             scaling_factor[1][p_idx])
            ub_p = ca.DM.zeros(n_p) + ub * scaling_factor[1][p_idx] if ub_p is None else ca.vertcat(ub_p,
                                                                                             ca.DM.zeros(n_p) + ub *
                                                                                             scaling_factor[1][p_idx])
        # init_p = ca.DM.zeros(p.shape) + p_init * scaling_factor[1]
        # lb_p = ca.DM.zeros(p.shape) + (p_init - 0.8) * scaling_factor[1]
        # ub_p = ca.DM.zeros(p.shape) + (p_init + 1) * scaling_factor[1]
        lbx = ca.vertcat(lbx, lb_p)
        ubx = ca.vertcat(ubx, ub_p)
        x0 = ca.vertcat(x0, init_p)
        x = ca.vertcat(x, p)
    if with_torque:
        lb_tau = ca.DM.ones(tau_0.shape[0]) * (-50) * scaling_factor[2]
        ub_tau = ca.DM.ones(tau_0.shape[0]) * 50 * scaling_factor[2]
        init_tau = tau_0
        #init_tau = ca.DM.zeros((tau.shape[0] - 4) * ns)
        lbx = ca.vertcat(lbx, lb_tau)
        ubx = ca.vertcat(ubx, ub_tau)
        x0 = ca.vertcat(x0, init_tau)
        x = ca.vertcat(x, pas_tau)
    return {"lbx": lbx, "ubx": ubx, "lbg": lbg, "ubg": ubg, "x0": x0, "x": x}


def _get_params_init(model, params_to_optim, use_ratio_tracking=False):
    model_param_init = []
    for p in params_to_optim:
        if p == "f_iso":
            model_param_init.append(
                [model.muscle(k).characteristics().forceIsoMax().to_mx() for k in range(model.nbMuscles())])
        elif p == "lm_optim":
            model_param_init.append(
                [model.muscle(k).characteristics().optimalLength().to_mx() for k in range(model.nbMuscles())])
        elif p == "lt_slack":
            model_param_init.append(
                [model.muscle(k).characteristics().tendonSlackLength().to_mx() for k in range(model.nbMuscles())])
    ratio = [model.muscle(k).characteristics().tendonSlackLength().to_mx() / model.muscle(
        k).characteristics().optimalLength().to_mx() for k in range(model.nbMuscles())]
    return model_param_init, ratio


def _get_passive_joint_torque(model, q, qdot, with_casadi=True):
    passive_torque = np.zeros((model.nbGeneralizedTorque(), q.shape[1]))
    if not with_casadi:
        for i in range(q.shape[1]):
            passive_torque[:, i] = model.passiveJointTorque(q[:, i], qdot[:, i]).to_array()
    if with_casadi:
        return ca.MX.zeros(model.nbGeneralizedTorque(), q.shape[1])

        def passive_torque_cas(model, q, qdot):
            return model.passiveJointTorque(q, qdot).to_mx()

        q_sym = ca.MX.sym("q", model.nbQ())
        qdot_sym = ca.MX.sym("qdot", model.nbQdot())
        mjt = passive_torque_cas(model, q_sym, qdot_sym)
        mjt_func = ca.Function("mjt_func", [q_sym, qdot_sym], [mjt]).expand()
        passive_torque = np.zeros((model.nbGeneralizedTorque(), q.shape[1]))
        for i in range(q.shape[1]):
            if with_casadi:
                passive_torque[:, i] = mjt_func(q[:, i], qdot[:, i])

    return passive_torque

def J_hes():
    model_ca = model
    if with_torque:
        hes_params = ca.vertcat(x, pas_tau, p_sym)
    else:
        hes_params = ca.vertcat(x, p_sym)
    jac_fct_f_all = ca.hessian(J, hes_params)[0]
    jac_fct_f = ca.Function("jac_fct_f", [hes_params], [jac_fct_f_all])
    x_tmp = ca.MX.ones(model_ca.nbMuscles() * ns) * 0.5 * scaling_factor[0]
    pas_tau_tmp = ca.MX.ones(len(passive_torque_idx) * ns) * 0.5 * scaling_factor[2]
    p_tmp = None
    for p_idx in range(len(p_mapping)):
        np_idx = len(p_mapping[p_idx][0])
        p_tmp = ca.MX.ones(np_idx) * scaling_factor[1][p_idx] if p_tmp is None else vertcat(p_tmp, ca.MX.ones(np_idx) *
                                                                                            scaling_factor[1][p_idx])
    if with_torque:
        hes_param_num = ca.vertcat(x_tmp, pas_tau_tmp, p_tmp)
    else:
        hes_param_num = ca.vertcat(x_tmp, p_tmp)
    jac_func_num = ca.Function("pouet", [], [jac_fct_f(hes_param_num)])()["o0"].toarray().squeeze()
    eigen_values = np.linalg.eigvals(jac_func_num)
    ev_max = min(eigen_values)
    ev_min = max(eigen_values)
    if ev_min == 0:
        condition_number = "! Ev_min is 0"
    if ev_min != 0:
        condition_number = np.abs(ev_max) / np.abs(ev_min)
    convexity = "Possible"
    for ev in range(eigen_values.size):
        if eigen_values[ev] < 0:
            convexity = "False"
            print("Convexity is false")
            break
    max_hes = np.ndarray.max(jac_func_num)
    min_hes = np.ndarray.min(jac_func_num)
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    fig_obj, axis_obj = plt.subplots(1, 1)
    jac_func_num[~(jac_func_num != 0).astype(bool)] = np.nan
    current_cmap3 = mcm.get_cmap("seismic")
    current_cmap3.set_bad(color="k")
    norm = mcolors.TwoSlopeNorm(vmin=min_hes - 0.01, vmax=max_hes + 0.01, vcenter=0)
    im3 = axis_obj.imshow(jac_func_num, cmap=current_cmap3, norm=norm)
    cbar_ax3 = fig_obj.add_axes([0.02, 0.4, 0.015, 0.3])
    fig_obj.colorbar(im3, cax=cbar_ax3)

    # plt.show()


def _perform_static_optim_parameters(emg_proc, biorbd_model_path, q, qdot, tau, muscle_track_idx,
                                     f_ext = None, use_p_mapping = False, with_param = True,
                                     mvc_normalized = True,
                                     with_torque = True,
                                     torque_as_constraint = False,
                                     dynamics_as_constraint = False,
                                     scaling_factor = (10, 10, 10),
                                     emg_init = None,
                                     muscle_list = None,
                                     p_mapping = None,
                                     use_casadi_fct = False,
                                     p_init = None,
                                     emg_names = None,
                                     params_to_optim = None,
                                     use_ratio_tracking = False,
                                     passive_torque_idx = None,
                                     param_bounds = None,
                                     state_int=None,
                                     all_muscle_len=None
                                     ) -> object:
    model = biorbd_ca.Model(biorbd_model_path)
    emg, emg_init = _map_activation(emg_proc, emg_init=emg_init, emg_names=emg_names,
                                    muscle_track_idx=muscle_track_idx,
                                    mvc_normalized=mvc_normalized,
                                    model=model)

    n_p = len(sum([i[0] for i in p_mapping], []))
    pas_tau = None
    p_sym = None
    passive_torque_idx = [i for i in range(model.nbQ() - 10,
                                           model.nbGeneralizedTorque())] if passive_torque_idx is None else passive_torque_idx
    pas_tau_sym = None
    # define casadi variables
    ns = q.shape[1]
    if with_param:
        p_sym = ca.MX.sym("p_sym", n_p)
    x_sym = ca.MX.sym("x_sym", model.nbMuscles())
    q_sym = ca.MX.sym("q_sym", model.nbQ())
    qdot_sym = ca.MX.sym("qdot_sym", model.nbQ())
    tau_sym = ca.MX.sym("tau_sym", tau.shape[0])
    f_ext_sym = ca.MX.sym("f_ext_sym", f_ext.shape[0])
    emg_sym = ca.MX.sym("emg_sym", emg.shape[0])
    if with_torque:
        pas_tau_sym = ca.MX.sym("pas_tau_sym", len(passive_torque_idx))

    # experimental data
    q = ca.MX(q)
    qdot = ca.MX(qdot)
    tau_mat = tau.copy()
    tau = ca.MX(tau)
    if f_ext is not None:
        f_ext = ca.MX(f_ext)
    #forward_function, next_x = _return_forward_function(model)
    l_norm_bounded = False
    #h = 0.008 / 4
    ca_funct = None
    mus_j_torque = None
    model_param_init, ratio_init = _get_params_init(model, params_to_optim, use_ratio_tracking)
    # passive_joint_torque = _get_passive_joint_torque(model, q, qdot)
    # param_list = _return_param_from_mapping(p_mapping, p_sym)
    # model = _apply_params(model, param_list, params_to_optim, model_param_init, ratio=ratio_init)
    if use_casadi_fct:
        ca_funct = _muscle_torque(model, scaling_factor,
                                  # , x_sym, q_sym, qdot_sym,
                                  p_mapping,
                                  # p_sym,
                                  muscle_list,
                                  use_p_mapping=use_p_mapping,
                                  with_param=with_param
                                  , return_casadi_function=True,
                                  params_to_optim=params_to_optim,
                                  model_params_init=model_param_init, ratio=ratio_init)
    else:
        mus_j_torque = _muscle_torque(model, scaling_factor,
                                      x, q, qdot,
                                      p_mapping, p_sym,
                                      muscle_list,
                                      use_p_mapping=use_p_mapping,
                                      with_param=with_param,
                                      return_casadi_function=False,
                                      params_to_optim=params_to_optim,
                                      model_params_init=model_param_init)
    weights = {"tau_tracking": 10,
               "activation_tracking": 80,
               "min_act": 10,
               "min_f_iso": 8,
               "min_lm_optim": 10,
               "min_lt_slack": 100,
               "min_pas_torque": 1,
               "ratio_tracking": 1,
               "dynamics": 100}

    # create n dependant function
    J = _get_cost_n_dependant(model, scaling_factor, x_sym, q_sym, qdot_sym, tau_sym, p_sym, pas_tau_sym, emg_sym, weights,
                           p_mapping=p_mapping,
                           with_torque=with_torque,
                           muscle_track_idx=muscle_track_idx,
                           muscle_casadi_function=ca_funct,
                           with_param=with_param,
                           passive_torque_idx=passive_torque_idx,
                           tau_as_constraint=torque_as_constraint)

    symlist = [x_sym, q_sym, qdot_sym, tau_sym, emg_sym]
    if with_torque:
        symlist.append(pas_tau_sym)
    if with_param:
        symlist.append(p_sym)
    J_func = ca.Function("J1", symlist, [J]).expand()
    J_mapped = J_func.map(ns, "thread", 6)
    # obj_1 = J_func(x_sym, q, qdot, tau, p_sym, pas_tau_sym, emg)
    x_all = ca.MX.sym("x_all", model.nbMuscles() * ns)
    x_split = ca.reshape(x_all, model.nbMuscles(), ns)
    p_all = ca.MX.sym("p_all", n_p)

    if with_torque:
        pas_tau_all = ca.MX.sym("pas_tau_all", len(passive_torque_idx) * ns)
        tau_split = ca.reshape(pas_tau_all, len(passive_torque_idx), ns)
        obj_1 = J_mapped(x_split, q, qdot, tau, emg, tau_split, ca.repmat(p_all, 1, ns))
    else:
        pas_tau_all=None
        obj_1 = J_mapped(x_split, q, qdot, tau, emg, ca.repmat(p_all, 1, ns))
    obj_1 = ca.sum2(obj_1)

    p_sym_2 = ca.MX.sym("p_sym_2", n_p)
    J_2, g = _add_params_to_J(p_sym_2, p_mapping, params_to_optim, scaling_factor, use_ratio_tracking, weights,
                              param_init=model_param_init,
                     ratio=ratio_init, bounds_l_norm=l_norm_bounded, muscle_len=all_muscle_len)
    J_2_func = ca.Function("J2", [p_sym_2], [J_2]).expand()
    if l_norm_bounded:
        g_fun = ca.Function("g", [p_sym_2], [g]).expand()
        g = ca.sum1(g_fun(p_all))
    obj_2 = ca.sum1(J_2_func(p_all))
    total_obj = obj_1 + obj_2
    total_obj /= 1000

    x0, tau_0 = _get_initial_values(model, passive_torque_idx, ns, muscle_track_idx, emg, scaling_factor)

    bounds_dic = _return_bounds(model, scaling_factor, p_all, ns, tau, x_all, pas_tau_all, x0, tau_0, with_param, with_torque,
                                p_init, params_to_optim, p_mapping, param_bounds, l_norm_bounded, torque_as_constraint, dynamics_as_constraint)
    #bounds_dic["x0"][0:ns * model.nbMuscles()] = x0
    opts = {"ipopt": {"max_iter": 1000, "print_level": 5, "linear_solver": "ma57",
                      "hessian_approximation": "exact", "acceptable_tol": 1e-2,
                      #"threads":4,
                      # "nlp_scaling_method": None,
                      #"linear_system_scaling": None,
                      # "fast_step_computation": "yes"
                      }}
    #ca.parallel.set_num_threads(4)
    if torque_as_constraint or g is not None:
        nlp = {"x": bounds_dic["x"], "f": total_obj, "g": g}
        sol_nlp = ca.nlpsol("sol", "ipopt", nlp, opts)
        solution = sol_nlp(x0=bounds_dic["x0"], lbx=bounds_dic["lbx"], ubx=bounds_dic["ubx"], lbg=bounds_dic["lbg"], ubg=bounds_dic["ubg"])
    else:
        nlp = {"x": bounds_dic["x"], "f": total_obj}
        sol_nlp = ca.nlpsol("sol", "ipopt", nlp, opts)
        solution = sol_nlp(x0=bounds_dic["x0"], lbx=bounds_dic["lbx"], ubx=bounds_dic["ubx"])
    # if with_param:
    #     print(solution["x"][model.nbMuscles() * (ns):-tau.shape[0]] / scaling_factor[1] + 1)
    #     print(solution["x"][-tau.shape[0]:] / scaling_factor[2])
    act = np.zeros((model.nbMuscles(), ns))
    pas_tau_mat = tau_mat.copy()
    if with_torque:
        sol_pas_tau = solution["x"][-len(passive_torque_idx) * ns:].toarray().squeeze()
    for j in range(ns):
        act[:, j] = np.array(solution["x"][
                             j * model.nbMuscles(): j * model.nbMuscles() + model.nbMuscles()].toarray().squeeze()) / \
                    scaling_factor[0]
        if with_torque:
            count = 0
            for k in passive_torque_idx:
                pas_tau_mat[k, j] = (np.array(
                    sol_pas_tau[j * len(passive_torque_idx): (j + 1) * len(passive_torque_idx)]) /
                                    scaling_factor[2])[count]
                count += 1
    n_passive_tau = len(passive_torque_idx)
    to_add = 0
    if with_torque:
        p = np.array(solution["x"][model.nbMuscles() * ns:-n_passive_tau * ns]).squeeze()

    else:
        p = np.array(solution["x"][model.nbMuscles() * ns:]).squeeze()
        pas_tau_mat = None
    p_tmp = None
    count = 0
    for p_idx, param in enumerate(params_to_optim):
        n_p = len(p_mapping[p_idx][0])
        p_tmp = p[count:count + n_p] / scaling_factor[1][p_idx] if p_tmp is None else ca.vertcat(p_tmp,
                                                                                                     p[
                                                                                                     count:count + n_p] /
                                                                                                     scaling_factor[1][
                                                                                                         p_idx])
        count += n_p
    p_list = _return_param_from_mapping(p_mapping, p_tmp)
    return act, pas_tau_mat, p_list, emg

def len_fct(model, q):
    #model.UpdateKinematicsCustom(q)
    #model.updateMuscles(q, True)
    mus_list = MX.zeros(model.nbMuscles(), 1)
    for i in range(model.nbMuscles()):
        #mus_list[i] = model.muscle(i).musculoTendonLength(model, q).to_mx()
        mus_list[i] = model.muscle(i).length(model, q).to_mx()
    return mus_list

def check_muscle_sanity(model_path, q, q_dot, plot_passive=True, plot_moment_arm=True, plot_length=True):
    model = biorbd.Model(model_path)
    moment_arm = np.zeros((model.nbMuscles(), q.shape[0], q.shape[1]))
    length = np.zeros((model.nbMuscles(), q.shape[1]))
    length_ca = np.zeros((model.nbMuscles(), q.shape[1]))

    mus_passive = np.zeros((model.nbMuscles(), q.shape[1]))
    init_f_iso = MX.zeros(model.nbMuscles())
    init_l_opt = MX.zeros(model.nbMuscles())
    for m in range(model.nbMuscles()):
        init_f_iso[m] = model.muscle(m).characteristics().forceIsoMax()
        init_l_opt[m] = model.muscle(m).characteristics().optimalLength()

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
    #     passive_torque_num[:, i] = (b1 * np.exp(k1 * (q[:, i]-qmid)) + b2 * np.exp(k2 * (q[:, i]-qmid))) * (1-deltap*0/(sv*wmax)) * (q[:, i] - deltap) + taueq
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
    mus_fpe = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_f_tot = np.zeros((model.nbMuscles(), q.shape[1]))
    mus_torque = np.zeros((model.nbGeneralizedTorque(), model.nbMuscles(), q.shape[1]))
    for i in range(q.shape[1]):
        passive_torque[:, i] = model.passiveJointTorque(q[:, i], np.zeros_like(q[:, i])).to_array()
        moment_arm[:, :, i] = -model.musclesLengthJacobian(q[:, i]).to_array()
        muscle_states = model.stateSet()
        for m in range(model.nbMuscles()):
            muscle_states[m].setActivation(1)
        #mus_torque[:, :, i] = model.muscularJointTorque(muscle_states, q[:, i], q_dot[:, i]).to_array()
        # length_ca[:, i] = cas_fct(ca.MX(q[:, i]))
        length_ca[:, i] = ca.Function("pouet", [MX()], [cas_fct(ca.MX(q[:, i]))])()["o0"].toarray().squeeze()
        for m in range(model.nbMuscles()):
            mus_tmp = biorbd.HillDeGrooteType(model.muscle(m))
            #model.UpdateKinematicsCustom(q[:, i])
            #model.updateMuscles(q[:, i], True)
            length[m, i] = model.muscle(m).length(model, q[:, i])  #
            mus_tmp.length(model, q[:, i])
            mus_tmp.velocity(model, q[:, i], q_dot[:, i], True)
            mus_tmp.computeFlPE()
            mus_tmp.computeFlCE(muscle_states[m])
            mus_tmp.computeFvCE()
            mus_fvce[m, i] = mus_tmp.FlCE(muscle_states[m]) * mus_tmp.characteristics().forceIsoMax()
            mus_fpe[m, i] = mus_tmp.FvCE() * mus_tmp.characteristics().forceIsoMax()
            mus_f_tot[m, i] = mus_tmp.characteristics().forceIsoMax() * (0.5 * mus_tmp.FlCE(muscle_states[m]) * mus_tmp.FvCE())
            mus_passive[m, i] = mus_tmp.FlPE() * mus_tmp.characteristics().forceIsoMax()
    # for i in range(model.nbMuscles()):
    #     max_ma = np.max(moment_arm[i, ...])
    #     max_moment_arm = np.where(moment_arm[i, ...] == max_ma)[0]
    #     print(f"max moment arm of {max_ma} for muscle {model.muscleNames()[i].to_string()} at index {model.nameDof()[int(max_moment_arm)].to_string()}")
    if plot_passive:
        plt.figure("passive_force")
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            plt.plot(mus_passive[i, :])
            plt.plot(mus_fvce[i, :], ".-")
            plt.plot(mus_fpe[i, :], "--")
            plt.plot(np.repeat(model.muscle(i).characteristics().forceIsoMax(), q.shape[1]), "k--")

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
            for i in range(6, model.nbDof()):
                plt.plot(moment_arm[j, i, :], label=model.nameDof()[i].to_string())
                plt.title(model.muscleNames()[j].to_string())
        plt.legend([name.to_string() for name in model.nameDof()][6:])
        plt.figure("torque")
        for j in range(model.nbMuscles()):
            plt.subplot(6, 7, j + 1)
            for i in range(6, model.nbDof()):
                plt.plot(moment_arm[j, i, :] * mus_passive[j, :], label=model.nameDof()[i].to_string())
                plt.plot(moment_arm[j, i, :] * mus_f_tot[j, :], ".-", label=model.nameDof()[i].to_string())

                plt.title(model.muscleNames()[j].to_string())
            plt.gca().set_prop_cycle(None)
        plt.legend([name.to_string() for name in model.nameDof()][6:])

    if plot_length:
        plt.figure("length")
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            plt.plot(length[i, :] / model.muscle(i).characteristics().optimalLength(), "r")
            #plt.plot(length_ca[i, :])
            plt.plot(np.repeat(model.muscle(i).characteristics().optimalLength()/ model.muscle(i).characteristics().optimalLength(), q.shape[1]), "k--")
            plt.title(model.muscleNames()[i].to_string())
        # plt.ylim([0, 1])
    plt.show()


def _apply_delay(q, qdot, tau, f_ext, emg_proc, em_delay, n_final, n_init, target_n_nodes, rate=120):
    em_delay_frame = int(em_delay * rate)
    emg_proc = emg_proc[:, n_init:-n_final]
    q, qdot, fext, tau = q[:, n_init:-n_final], qdot[:, n_init:-n_final], f_ext[:, n_init:-n_final], tau[:,
                                                                                                     n_init:-n_final]
    q, qdot, fext, tau = q[:, em_delay_frame:], qdot[:, em_delay_frame:], fext[:, em_delay_frame:], tau[:,
                                                                                                    em_delay_frame:]
    emg_proc = emg_proc[:, :-em_delay_frame] if em_delay_frame > 0 else emg_proc
    down_sampling = (q.shape[1]-1) // (target_n_nodes)
    q_init, qdot_init, fext_init, tau_init = q.copy(), qdot.copy(), fext.copy(), tau.copy()
    q, qdot, fext, tau = q[:, :-1:down_sampling], qdot[:, :-1:down_sampling], fext[:, :-1:down_sampling], tau[:,
                                                                                                    :-1:down_sampling]
    emg_proc = emg_proc[:, :-1:down_sampling]
    q_int = np.zeros_like(q)
    q_dot_int = np.zeros_like(qdot)
    for i in range(q.shape[1]):
        idx = np.where(q_init == q[:, i][:, None])[1][0]
        q_int[:, i] = q_init[:, idx+1]
        q_dot_int[:, i] = qdot_init[:, idx+1]

    state_int = np.concatenate((q_int, q_dot_int), axis=0)
    return q, qdot, tau, fext, emg_proc, state_int


def _get_ratio(model, use_casadi=True):
    ratio = [model.muscle(k).characteristics().tendonSlackLength() / model.muscle(
        k).characteristics().optimalLength() for k in range(model.nbMuscles())]
    return ratio


def main():
    muscle_list = [
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
    # emg_names = ["PECM",
    #              "bic",
    #              "tri",
    #              "LAT",
    #              'TRP1'
    #              "DELT1",
    #              'DELT2',
    #              'DELT3'
    #              ]

    muscle_list = ['TrapeziusScapula_M',  # 0
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

    # emg_names = ["PectoralisMajorThorax",
    #              "BIC",
    #              "TRI",
    #              #"LatissimusDorsi",
    #              'TrapeziusScapula_S',
    #              #'TrapeziusClavicle',
    #              "DeltoideusClavicle_A",
    #              'DeltoideusScapula_M',
    #              'DeltoideusScapula_P']
    #
    #
    # muscle_track_idx = []
    # for i in range(len(emg_names)):
    #     muscle_track_idx.append([j for j in range(len(muscle_list)) if emg_names[i] in muscle_list[j]])
    # muscle_track_idx = sum(muscle_track_idx, [])

    #muscle_list = [model.muscleNames()[i].to_string() for i in range(model.nbMuscles())]
    from biosiglive import load
    from math import ceil
    import matplotlib.pyplot as plt
    perform_optim = False
    # sensix_data = load(f"data/active_global_ref.bio")
    # f_ext = -np.array([sensix_data["LMY"],
    #                    -sensix_data["LMX"],
    #                    sensix_data["LMZ"],
    #                    sensix_data["LFY"],
    #                    -sensix_data["LFX"],
    #                    sensix_data["LFZ"]])
    # f_ext[:3, :] = np.zeros((3, f_ext.shape[1]))
    # import biorbd_casadi as biorbd_ca
    # start = kalman_data["q"].shape[0] - biorbd_ca.Model(biorbd_model_path).nbQ()
    # kalman = np.concatenate((kalman_data["q"][start:, 108:3737], kalman_data["q_dot"][start:, 108:3737]), axis=0)
    # kalman_data["q"] = kalman_data["q"][start:, 108:3737]
    # kalman_data["q_dot"] = kalman_data["q_dot"][start:, 108:3737]
    # kalman_data = load(f"data/P3_gear_20_kalman.bio")
    # data_dir = f"data/"
    # trials = [
    #     "init_kalman_data_flex_poid_2kg.bio",
    #     "init_kalman_data_abd_poid_2kg.bio",
    #     "init_kalman_data_cycl_poid_2kg.bio",
    # ]
    part = "P10"
    file_name = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}"
    all_dir = os.listdir(file_name)
    trial = [dir for dir in all_dir if "gear_20" in dir][0]

    model = "normal_500_down_b1"
    # trials = [f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial}/reoriented_dlc_markers.bio"]
    prefix = "/mnt/shared"
    trials = [
        prefix + f"/Projet_hand_bike_markerless/process_data/{part}/result_biomech_{trial.split('_')[0]}_{trial.split('_')[1]}_{model}_no_root.bio"]
    source = "dlc_1"
    biorbd_model_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/gear_20_model_scaled_{source[:-2]}_ribs_new_seth.bioMod"
    file_name = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial}/result_{trial.split('/')[-1]}.bio"
    #biorbd_model_path = "/mnt/shared/Projet_hand_bike_markerless/RGBD/P10/wu_bras_gauche_depth.bioMod"
    if perform_optim:
        for trial in trials:
            kalman_data = load(trial)
            n_start = 300  #int(7) + 8
            n_stop = 600  #int(390) - 157
            with_mhe = True
            final_time = (n_stop - n_start - 1) / 120 if not with_mhe else 0.1
            n_shooting = n_stop - n_start - 1 if not with_mhe else int(final_time * 120)
            with_mhe = True
            with_f_ext = True
            track_previous = False
            f_ext = kalman_data["f_ext"][:, n_start:n_stop] if with_f_ext else None
            if with_f_ext:
                if part == "P14":
                    f_ext = -f_ext
            q_init = kalman_data[source]["q_raw"][:, n_start:n_stop]
            q_dot_init = kalman_data[source]["q_dot"][:, n_start:n_stop]
            model = biorbd.Model(biorbd_model_path)
            names = kalman_data[source]["marker_names"]
            ia_idx = names.index("SCAP_IA")
            ts_idx = names.index("SCAP_TS")
            mark_ia = kalman_data[source][f"tracked_markers"][:, ia_idx, :].copy()
            mark_ts = kalman_data[source][f"tracked_markers"][:, ts_idx, :].copy()
            kalman_data[source][f"tracked_markers"][:, ia_idx, :] = mark_ts
            kalman_data[source][f"tracked_markers"][:, ts_idx, :] = mark_ia
            markers_init = kalman_data[source][f"tracked_markers"][:, :, n_start:n_stop]

            # markers_init[:, 1, :] = np.repeat(markers_init[:, 1, 0][:, np.newaxis], markers_init.shape[2], axis=1)
            if model.nbDof() > q_init.shape[0]:
                q_init = np.concatenate((np.zeros((model.nbQ() - 10, q_init.shape[1])), q_init), axis=0)
                q_init = np.concatenate((q_init, np.zeros((model.nbQ() - 10, q_init.shape[1])), q_dot_init), axis=0)
            else:
                q_init = np.concatenate((q_init, q_dot_init), axis=0)
            # q_init = np.concatenate((q_init[3:, :], q_dot_init[3:, :]), axis=0)
            ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
                              final_time=final_time,
                              n_shooting=n_shooting,
                              use_sx=True,
                              n_threads=1,
                              with_muscle=False,
                              target=markers_init,
                              q_init=q_init,
                              f_ext=f_ext,
                              with_f_ext=with_f_ext,
                              mhe=with_mhe,
                              track_previous=track_previous,
                              # params=np.ones((6,1))
                              )
            if with_mhe:
                def update_functions(mhe, t, _):
                    def target_mark(i: int):
                        # return kalman_data["q"][:, n_start:n_stop][:, i : i + n_shooting + 1]
                        return markers_init[:, :, i: i + n_shooting + 1]

                    def target_f_ext(i: int):
                        return f_ext[:, i: i + n_shooting + 1]

                    if with_f_ext:
                        mhe.update_objectives_target(target=target_f_ext(t), list_index=2)
                        mhe.update_objectives_target(target=target_mark(t)[:, :3, :], list_index=3)
                        mhe.update_objectives_target(target=target_mark(t)[:, 3:, :], list_index=4)
                        if track_previous:
                            q_to_track = ocp.sol.decision_states(to_merge=SolutionMerge.NODES)[
                                "q"] if ocp.sol is not None else q_init[:model.nbQ(), t:t + n_shooting + 1]
                            qdot_to_track = ocp.sol.decision_states(to_merge=SolutionMerge.NODES)[
                                "qdot"] if ocp.sol is not None else q_init[model.nbQ():, t:t + n_shooting + 1]
                            mhe.update_objectives_target(target=q_to_track, list_index=7)
                            mhe.update_objectives_target(target=qdot_to_track, list_index=8)
                    else:
                        mhe.update_objectives_target(target=target_mark(t)[:, :3, :], list_index=2)
                        mhe.update_objectives_target(target=target_mark(t)[:, 3:, :], list_index=3)
                        if track_previous:
                            q_to_track = ocp.sol.decision_states(to_merge=SolutionMerge.NODES)[
                                "q"] if ocp.sol is not None else q_init[:model.nbQ(), t:t + n_shooting + 1]
                            qdot_to_track = ocp.sol.decision_states(to_merge=SolutionMerge.NODES)[
                                "qdot"] if ocp.sol is not None else q_init[model.nbQ():, t:t + n_shooting + 1]
                            mhe.update_objectives_target(target=q_to_track, list_index=6)
                            mhe.update_objectives_target(target=qdot_to_track, list_index=7)
                    return t < kalman_data[source]["q_raw"][:, n_start:n_stop].shape[1] - (
                            n_shooting + 1)  # True if there are still some frames to reconstruct
                    # return t < 15 # True if there are still some frames to reconstruct

                sol = ocp.solve(update_functions, **get_solver_options(Solver.ACADOS()))
            else:
                # Solve the program
                solver = Solver.IPOPT()
                solver.set_linear_solver("ma57")
                solver.set_hessian_approximation("exact")
                solver.set_tol(1e-5)
                solver.set_maximum_iterations(1000)
                sol = ocp.solve(solver=solver)
            # --- Solve the program using ACADOS --- #
            #
            from biosiglive import save
            from bioptim import SolutionMerge
            merged_states = sol.decision_states(to_merge=SolutionMerge.NODES)
            merged_controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
            save_dic = {"q": merged_states["q"],
                        "qdot": merged_states["qdot"],
                        "tau": merged_controls["tau"],
                        "init_q": kalman_data[source]["q_raw"][:, n_start:n_stop - (n_shooting + 1)],
                        "init_qdot": kalman_data[source]["q_dot"][:, n_start:n_stop - (n_shooting + 1)],
                        "init_markers": markers_init[:, :, :-(n_shooting + 1)],
                        "emg": kalman_data["emg"][:, n_start:n_stop - (n_shooting + 1)],
                        "peaks": kalman_data["peaks"],
                        "init_f_ext": None,
                        "mhe": with_mhe,
                        "n_shooting": n_shooting,
                        "mhe_time": final_time,
                        "n_start": n_start,
                        "n_stop": n_stop,
                        }
            if with_f_ext:
                save_dic["f_ext"] = merged_controls["f_ext"]
                save_dic["init_f_ext"] = f_ext[:, :-(n_shooting + 1)]

            # if with_f_ext:
            #     save_dic["fext"] = sol.controls["f_ext"],

            save(save_dic, file_name,
                 safe=False)
            # integrated_sol = sol.integrate()
            #sol.graphs()

    trial_to_concatenate = [file_name]
    q_conc, qdot_conc, tau_conc, emg_conc, f_ext_conc = None, None, None, None, None
    for t, trial in enumerate(trial_to_concatenate):
        # sensix_data = load(f"data/P3_gear_{trial}_sensix.bio")
        # f_ext = np.array([sensix_data["RMY"],
        #                   -sensix_data["RMX"],
        #                   sensix_data["RMZ"],
        #                   sensix_data["RFY"],
        #                   -sensix_data["RFX"],
        #                   sensix_data["RFZ"]])
        from biosiglive import OfflineProcessing, OfflineProcessingMethod
        # trials_init = [
        #     "data_flex_poid_2kg.bio",
        #     "data_abd_poid_2kg.bio",
        #     "data_cycl_poid_2kg.bio",
        # ]
        ocp_result = load(trial)
        q = ocp_result["q"]
        qdot = ocp_result["qdot"]
        tau = ocp_result["tau"]
        emg_proc = ocp_result["emg"]
        f_ext = ocp_result["f_ext"]
        q_init = ocp_result["init_q"]

        import bioviz
        model = biorbd.Model(biorbd_model_path)
        f_ext_mat = np.zeros((1, 6, f_ext.shape[1]))
        for i in range(f_ext.shape[1]):
            B = [0, 0, 0, 1]
            all_jcs = model.allGlobalJCS(q[:, i])
            RT = all_jcs[-1].to_array()
            B = RT @ B
            vecteur_OB = B[:3]
            f_ext_mat[0, :3, i] = vecteur_OB
            #f_ext_mat[0, :3, i] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])
            f_ext_mat[0, 3:, i] = f_ext[3:, i]
        # q[6:11, :] = np.zeros_like(q[6:11, :])
        b = bioviz.Viz(loaded_model=model, mesh_opacity=1)
        b.load_movement(q)
        b.load_experimental_markers(ocp_result["init_markers"])
        b.load_experimental_forces(f_ext_mat, segments=["ground"], normalization_ratio=0.5)
        b.exec()

        # if q_init.shape != q.shape:
        #     q_init = np.concatenate((np.zeros((q.shape[0] - 10, q_init.shape[1])), q_init), axis=0)
        #     q_dot_init = np.concatenate((np.zeros((qdot.shape[0] - 10, q_init.shape[1])), ocp_result["init_qdot"]),
        #                                 axis=0)

        plt.figure("angle")
        for i in range(q.shape[0]):
            plt.subplot(ceil(q.shape[0] / 4), 4, i + 1)
            plt.plot(q_init[i, :], "r", )
            plt.plot(q[i, :])
        #
        # plt.figure("angle_crank")
        # plt.plot(sensix_data["crank_angle"][n_start:n_stop])

        plt.figure("vitesse")
        for i in range(q.shape[0]):
            plt.subplot(ceil(q.shape[0] / 4), 4, i + 1)
            plt.plot(q_init[q.shape[0] + i:, :], "r", )
            plt.plot(ocp_result["qdot"][i, :])

        plt.figure("tau")
        for i in range(q.shape[0]):
            plt.subplot(ceil(q.shape[0] / 4), 4, i + 1)
            plt.plot(ocp_result["tau"][i, :])
            # plt.plot(tau_proc[i, :])

        # plt.figure("emg")
        # for i in range(emg.shape[0]):
        #     plt.subplot(ceil(emg.shape[0] / 4), 4, i + 1)
        #     plt.plot(emg_proc[i, :])
        #     plt.title(emg_names[i])

        plt.figure("force")
        for i in range(f_ext.shape[0]):
            plt.subplot(ceil(f_ext.shape[0] / 4), 4, i + 1)
            plt.plot(ocp_result["init_f_ext"][i, :], "r", label="sensix", )
            plt.plot(f_ext[i, :])
        # plt.legend()

        # plt.figure("markers")
        # for i in range(kalman_data["markers"].shape[1]):
        #     plt.subplot(ceil(kalman_data["markers"].shape[1]/4), 4, i+1)
        #     plt.plot(kalman_data["markers"][1, i, n_start:n_stop])
        #     plt.plot(kalman_data["markers"][2, i, n_start:n_stop])
        #     plt.plot(kalman_data["markers"][0 , i, n_start:n_stop])
        #plt.show()
        import biorbd_casadi as biorbd_ca

        # forward_function, next_x = _return_forward_function(biorbd_ca.Model(biorbd_model_path))
        em_delay = 0.04
        n_final = 100
        n_init = 30
        target_n_nodes = 50

        #biorbd_model_path = "/mnt/shared/Projet_hand_bike_markerless/RGBD/P10/model_scaled_depth_new_test_param.bioMod"
        #biorbd_model_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/P10/wu_bras_gauche_seth_for_cycle_param.bioMod"

        biorbd_model = biorbd_ca.Model(biorbd_model_path)
        q, qdot = q[:, :-1], qdot[:, :-1]
        q, qdot, tau, f_ext, emg_proc, state_int = _apply_delay(q, qdot, tau, f_ext, emg_proc, em_delay, n_final, n_init,
                                                     target_n_nodes)
        # emg_proc = np.delete(emg_proc, 3, axis=0)
        if emg_conc is None:
            emg_conc = emg_proc
            q_conc = q
            qdot_conc = qdot
            tau_conc = tau
        else:
            emg_conc = np.concatenate((emg_conc, emg_proc), axis=1)
            q_conc = np.concatenate((q_conc, q), axis=1)
            qdot_conc = np.concatenate((qdot_conc, qdot), axis=1)
            tau_conc = np.concatenate((tau_conc, tau), axis=1)

    emg_proc, q, qdot, tau = emg_conc, q_conc, qdot_conc, tau_conc
    with_param = True
    with_torque = True
    use_p_mapping = 0  # 0 = no, 1 = mapping, 2 = mapping as constrain
    use_ratio_tracking = True
    torque_as_constraint = False

    optim_param_list = ["f_iso",
                        "lm_optim",
                        ]  # ["f_iso", "lt_slack"]
    # q = q[6:, :]
    # qdot = qdot[6:, :]
    # tau = tau[6:, :]
    param_bounds = [[0, 1] for _ in optim_param_list]
    p_init = [1]
    all_muscle_len = None
    for p_idx, param in enumerate(optim_param_list):
        if param == "f_iso":
            param_bounds[p_idx] = [0.4, 2.8] #[0.5, 5]
        if param == "lm_optim":
            eigen_model = biorbd.Model(biorbd_model_path)
            all_muscle_len = _get_all_muscle_len(eigen_model, q)
            param_bounds[p_idx] = [0.4, 2.8] # [0.2, 2.8]
        if param == "lt_slack":
            param_bounds[p_idx] = [0.5, 1.5] #[0.5, 2.5]

    p_mapping = [list(range(biorbd_model.nbMuscles())), list(range(biorbd_model.nbMuscles()))]
    p_mapping_list = [p_mapping] * len(optim_param_list)
    list_mapping = list(range(biorbd_model.nbMuscles()))
    if use_p_mapping and "f_iso" in optim_param_list:
        muscle = muscle_list[0]
        all_idx = []
        unique_names = []
        for m in muscle_list:
            if m.split("_")[0] not in unique_names:
                unique_names.append(m.split("_")[0])
        for muscle in unique_names:
            n_repeat = 0
            for m in muscle_list:
                if muscle in m:
                    n_repeat += 1
            all_idx.append([unique_names.index(muscle)] * n_repeat)
        list_mapping = sum(all_idx, [])
        # list_mapping = [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11, 12, 12, 13, 14, 14, 14, 15, 15, 16, 17, 18, 18,
        #                 18, 19, 19, 20, 20, 20, 21, 21]
        # list_mapping = [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11, 12, 12, 13, 14, 14, 14, 15, 15, 16, 17, 18, 18,
        #                 18, 19, 19, 20, 21, 21]
        # list_mapping = [0, 1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 11, 11, 12, 13, 13, 13, 14, 14, 15, 16, 17, 17,
        #                 17, 18, 18, 19, 19, 19, 20, 20]
        p_mapping = [list(range(max(list_mapping) + 1)), list_mapping]
        p_mapping_list[optim_param_list.index("f_iso")] = p_mapping
    # q[6:11, :] = np.zeros_like(q[6:11, :])
    # qdot[6:11, :] = np.zeros_like(q[6:11, :])

    #check_muscle_sanity(biorbd_model_path, q, qdot, plot_passive=True, plot_moment_arm=True, plot_length=True)

    import time
    tic = time.time()
    perform_static_optim = True

    emg_names = ["PectoralisMajorThorax",
                 "BIC",
                 "TRI",
                 "LatissimusDorsi",
                 'TrapeziusScapula_S',
                 #'TrapeziusClavicle',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                  'DeltoideusScapula_P']
    # emg_names = ["PECM1",
    #              "bic",
    #              "tri",
    #              "LAT",
    #              'TRP2',
    #              #'TrapeziusClavicle',
    #              "DEL1",
    #              'DELT2',
    #              'DELT3']

    muscle_list = [name.to_string() for name in biorbd_model.muscleNames()]
    muscle_track_idx = []
    for i in range(len(emg_names)):
        muscle_track_idx.append([j for j in range(len(muscle_list)) if emg_names[i] in muscle_list[j]])
    muscle_track_idx = sum(muscle_track_idx, [])
    #passive_torque_idx = [3, 5, 6, 7, 8, 9, 10, 11]
    # optim_passive_torque = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # no_root
    optim_passive_torque = [i for i in range(q.shape[0] - 10, q.shape[0])]
    #tau[-1, :] = np.zeros((1, tau.shape[1]))
    if perform_static_optim:
        for i in p_init:
            scale_params = [1] * len(param_bounds)
            # scale_params[0] = 5
            scaling_factor = (1, scale_params, 1)
            # q[6:11, :] = np.zeros_like(q[6:11, :])
            # qdot[6:11, :] = np.zeros_like(q[6:11, :])
            # tau[6:11, :] = np.zeros_like(q[6:11, :])

            a, pas_tau, p, emg = _perform_static_optim_parameters(emg_proc, biorbd_model_path, q, qdot, tau,
                                                                  f_ext=f_ext,
                                                                  muscle_track_idx=muscle_track_idx,
                                                                  use_p_mapping=use_p_mapping,
                                                                  with_param=with_param,
                                                                  emg_init=emg_proc,
                                                                  mvc_normalized=True,
                                                                  with_torque=with_torque,
                                                                  torque_as_constraint=torque_as_constraint,
                                                                  dynamics_as_constraint=False,
                                                                  scaling_factor=scaling_factor,
                                                                  muscle_list=muscle_list,
                                                                  p_mapping=p_mapping_list,
                                                                  use_casadi_fct=True,
                                                                  p_init=i,
                                                                  emg_names=emg_names,
                                                                  params_to_optim=optim_param_list,
                                                                  use_ratio_tracking=use_ratio_tracking,
                                                                  passive_torque_idx=optim_passive_torque,
                                                                  param_bounds=param_bounds,
                                                                  state_int=state_int,
                                                                  all_muscle_len=all_muscle_len,
                                                                  )
            from biosiglive import save
            save(
                {"a": a, "pas_tau": pas_tau, "p": p, "emg": emg, "q": q, "qdot": qdot, "scaling_factor": scaling_factor,
                 "p_mapping": list_mapping, "p_init": i, "solving_time": time.time() - tic, "optimized_params": optim_param_list,
                 "param_bounds": param_bounds},
                f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param.bio",
                safe=False)

    data_list = load(
        f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param.bio",
        merge=False)

    for k in range(len(p_init)):
        data_param = data_list[k]
        a, pas_tau, p, emg = data_param["a"], data_param["pas_tau"], data_param["p"], data_param["emg"]
        print("param:", p)
        model = biorbd.Model(biorbd_model_path)
        mus_j_torque = np.zeros((tau.shape[0], tau.shape[1]))
        mus_torque = np.zeros((model.nbMuscles(), tau.shape[0], tau.shape[1]))
        mus_name_list = [name.to_string() for name in model.muscleNames()]
        moment_arm = np.zeros((model.nbMuscles(), q.shape[0], q.shape[1]))
        muscle_force = np.zeros((model.nbMuscles(), q.shape[1]))
        # model.muscle(m).characteristics().optimalLength() + (p_tmp[m]))
        ratio = None
        # plt.figure(f"tau_comparison")
        # model_tmp = model
        # for j in range(2):
        #     if j == 0 and "lt_slack" not in optim_param_list:
        #         ratio = _get_ratio(model_tmp, use_casadi=False)
        #     if j == 1:
        #         model_tmp = _apply_params(model_tmp, p, optim_param_list, with_casadi=False, ratio=ratio)
        #     for k in range(tau.shape[1]):
        #         muscle_states = model_tmp.stateSet()
        #         for m in range(model_tmp.nbMuscles()):
        #             muscle_states[m].setActivation(a[m, k])
        #         muscles_force = model_tmp.muscleForces(muscle_states, q[:, k], qdot[:, k]).to_array()
        #         muscle_force[:, k] = muscles_force
        #         moment_arm[:, :, k] = model_tmp.musclesLengthJacobian(q[:, k]).to_array()
        #         for m in range(model_tmp.nbMuscles()):
        #             mus_torque[m, :, k] = -moment_arm[m, :, k] * muscles_force[m]
        #         mus_j_torque[:, k] = model_tmp.muscularJointTorque(muscle_states, q[:, k], qdot[:, k]).to_array()
        #     for i in range(0, model.nbQ()):
        #         plt.subplot(5, 4, i + 1)
        #         plt.plot(tau[i, :], "-", color="r", label="tau_init")
        #         if with_torque:
        #             plt.plot(mus_j_torque[i, :] + pas_tau[i, :], color="b", label="optim_param")
        #             plt.plot(pas_tau[i, :], ".-", label="pas_torque", alpha=0.5, color="g")
        #             # plt.plot(passive_joint_torques[i, :], ".-", label="passive_joint_torque", alpha=0.5)
        #         plt.plot(mus_j_torque[i, :], "--", color="k", label="mus_j_torque", alpha=0.5)
        #         plt.title(model.nameDof()[i].to_string())

        ratio = None
        if with_param:
            for j in range(2):
                if j == 0 and "lt_slack" not in optim_param_list:
                    ratio = _get_ratio(model, use_casadi=False)
                if j == 1:
                    model = _apply_params(model, p, optim_param_list, with_casadi=False, ratio=ratio)
                for p_idx, param in enumerate(optim_param_list):
                    plt.figure(f"{param}")
                    p_tmp = p[p_idx]
                    for i in range(model.nbMuscles()):
                        to_plot = None
                        to_plot_ref = None
                        color = "g" if param_bounds[p_idx][0] + 0.05 < p_tmp[i] < param_bounds[p_idx][1] - 0.05 else "r"
                        if param == "f_iso":
                            #to_plot = model.muscle(i).characteristics().forceIsoMax() * float((p_tmp[i]))
                            to_plot_ref = model.muscle(i).characteristics().forceIsoMax()
                        elif param == "lm_optim":
                            #to_plot = model.muscle(i).characteristics().optimalLength() * float((p_tmp[i]))
                            to_plot_ref = model.muscle(i).characteristics().optimalLength()
                        elif param == "lt_slack":
                            #to_plot = model.muscle(i).characteristics().tendonSlackLength() * float((p_tmp[i]))
                            to_plot_ref = model.muscle(i).characteristics().tendonSlackLength()
                        #plt.bar(i, to_plot, width=0.25, color=color)
                        idx = i if j == 0 else i + 0.25
                        c = "b" if j == 0 else color
                        plt.bar(idx, to_plot_ref, width=0.25, color=c)
                        if j == 1:
                            plt.text(i, to_plot_ref, float(np.round(p_tmp[i], 2)[0][0]), ha='center', Bbox = dict(facecolor = 'white', alpha =.8), fontsize=8)
                    plt.title(f"bounds = {param_bounds[p_idx]}")
                    plt.xticks(np.arange(len(mus_name_list)), mus_name_list, rotation=90)
                    plt.legend(["optim", "init"])
        #optim_param_list = []
        for k in range(tau.shape[1]):
            muscle_states = model.stateSet()
            for m in range(model.nbMuscles()):
                muscle_states[m].setActivation(a[m, k])
            muscles_force = model.muscleForces(muscle_states, q[:, k], qdot[:, k]).to_array()
            muscle_force[:, k] = muscles_force
            moment_arm[:, :, k] = model.musclesLengthJacobian(q[:, k]).to_array()
            for m in range(model.nbMuscles()):
                mus_torque[m, :, k] = -moment_arm[m, :, k] * muscles_force[m]
            mus_j_torque[:, k] = model.muscularJointTorque(muscle_states, q[:, k], qdot[:, k]).to_array()
        passive_joint_torques = _get_passive_joint_torque(model, q, qdot, with_casadi=False)
        #plt.figure(f"tau_muscles")
        for i in range(0, model.nbQ()):
            #plt.subplot(6, 3, i + 1)
            plt.figure(f"tau_muscles_{model.nameDof()[i].to_string()}")
            plt.plot(tau[i, :], ".-", color="r", label="tau_init")
            if with_torque:
                plt.plot(mus_j_torque[i, :] + pas_tau[i, :], ".-", color="k",  label="optim_param")
                plt.plot(pas_tau[i, :], ".-", color="g", label="pas_torque")
                #plt.plot(passive_joint_torques[i, :], ".-", label="passive_joint_torque", alpha=0.5)
            plt.plot(mus_j_torque[i, :], "--",color="b", label="mus_j_torque")
            for m in range(model.nbMuscles()):
                if abs(mus_torque[m, i, :]).max() > 0.1:
                    plt.plot(mus_torque[m, i, :], alpha=0.5, label=model.muscleNames()[m].to_string())
            plt.gca().set_prop_cycle(None)
            plt.legend()

        plt.figure(f"tau_optim")
        for i in range(0, model.nbQ()):
            plt.subplot(5, 4, i + 1)
            plt.plot(tau[i, :], ".-", color="r", label="tau_init")
            if with_torque:
                plt.plot(mus_j_torque[i, :] + pas_tau[i, :], label="optim_param")
                plt.plot(pas_tau[i, :], ".-", label="pas_torque", alpha=0.5)
                #plt.plot(passive_joint_torques[i, :], ".-", label="passive_joint_torque", alpha=0.5)
            plt.plot(mus_j_torque[i, :], "--", label="mus_j_torque", alpha=0.5)
            plt.title(model.nameDof()[i].to_string())
        # plt.legend()

        plt.figure("act")
        count = 0
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            if i in muscle_track_idx:
                plt.plot(emg[muscle_track_idx.index(i), :], label="act_init", color="b")
            plt.plot(a[i, :], label="optim_param", color="g")
            plt.title(mus_name_list[i])
            plt.ylim([0, 1])
            # plt.plot(act[i, :])
        plt.legend(["act_init", "optim_param"])

        plt.figure("mus_force")
        count = 0
        for i in range(model.nbMuscles()):
            plt.subplot(6, 7, i + 1)
            plt.plot(muscle_force[i, :], label="optim_param", color="g")
            plt.title(mus_name_list[i])
            # plt.plot(act[i, :])
        #
        # plt.figure("q")
        # for i in range(q_int.shape[0]):
        #     plt.subplot(5, 2, i + 1)
        #     # plt.plot(q_int[i, 1:], label="q_int")
        #     # plt.scatter(0, q_int[i, :1])
        #     # plt.scatter(0, data_list[0]["q"][i, :1])
        #     plt.plot(q[i, :], label="q_init")
        #     plt.legend()
        # plt.figure("q_dot")
        # for i in range(q_int.shape[0]):
        #     plt.subplot(5, 2, i + 1)
        #     plt.plot(q_dot_int[i, :])
        #     plt.plot(qdot[i, :])

    plt.show()


if __name__ == "__main__":
    main()
