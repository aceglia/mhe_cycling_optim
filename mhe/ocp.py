"""
This code provide every function needed to solve the OCP problem.
"""
import bioptim
from .utils import *
from time import time, sleep
import biorbd_casadi as biorbd
import numpy as np
from casadi import MX, Function, horzcat, vertcat, mtimes, cross
from bioptim.misc.enums import SolverType
from bioptim.optimization.receding_horizon_optimization import RecedingHorizonOptimization
from bioptim import (

    MovingHorizonEstimator,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
DynamicsFcn,
    InitialGuessList,
    BoundsList,
    InterpolationType,
    SolutionMerge,
    Solver,
    Node,
    OptimalControlProgram,
    DynamicsFunctions,
    DynamicsEvaluation,
    BiorbdModel,
    NonLinearProgram,
    ConfigureProblem,
    ConstraintList,
    ConstraintFcn,
)


def define_constraint(f_ext_target: np.ndarray, with_f_ext: bool, f_ext_as_constraints: bool):
    constraints = ConstraintList()
    if with_f_ext and f_ext_as_constraints:
        constraints.add(ConstraintFcn.TRACK_CONTROL, key="f_ext", target=f_ext_target[0, :, :].T, node=Node.ALL_SHOOTING)
    return constraints


def define_objective(
    weights: dict,
    use_torque: bool,
    with_f_ext: bool,
    f_ext_as_constraints: bool,
    track_emg: bool,
    muscles_target: np.ndarray,
    f_ext_target : np.array,
    kin_target: np.ndarray,
    biorbd_model: BiorbdModel,
    previous_sol: np.ndarray,
    kin_data_to_track: str = "markers",
    muscle_track_idx: list = (),
):
    """
    Define the objective function of the OCP.

    Parameters
    ----------
    weights : dict
        Weights of the different terms.
    use_torque : bool
        If True, use the torque are used in the dynamics.
    track_emg : bool
        If True, track the EMG are tracked.
    muscles_target : np.ndarray
        Target of the muscles.
    kin_target : np.ndarray
        Target for kinematics objective.
    previous_sol : np.ndarray
        solution of the previous subproblem
    biorbd_model : BiorbdModel
        Model of the system.
    kin_data_to_track : str
        Kind of kinematics data to track ("markers" or "q").
    muscle_track_idx : list
        Index of the muscles to track.

    Returns
    -------
    Objective function.
    """
    previous_q, previous_qdot = (
        previous_sol[: biorbd_model.nb_q, :],
        previous_sol[biorbd_model.nb_q: biorbd_model.nb_q * 2, :],
    )
    if track_emg:
        muscle_min_idx = []
        for i in range(biorbd_model.nb_muscles):
            if i not in muscle_track_idx:
                muscle_min_idx.append(i)
    else:
        muscle_min_idx = np.array(range(biorbd_model.nb_muscles))

    objectives = ObjectiveList()
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_q"],
        index=np.array(range(biorbd_model.nb_q)),
        key="q",
        node=Node.ALL,
        multi_thread=False,
        quadratic=True,

    )
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_dq"],
        index=np.array(range(biorbd_model.nb_q)),
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
        quadratic=True,

    )
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=weights["previous_q"],
        target=previous_q[:, :],
        key="q",
        node=Node.ALL,
        multi_thread=False,
        quadratic=True,

    )
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=weights["previous_q_dot"],
        target=previous_qdot[:, :],
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
        quadratic=True,
    )

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weights["min_activation"],
        key="muscles",
        index=muscle_min_idx,
        multi_thread=False,
        quadratic=True,

    )

    if with_f_ext and f_ext_as_constraints is False:
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_CONTROL,
            weight=weights["f_ext"],
            target=f_ext_target,
            key="f_ext",
            node=Node.ALL_SHOOTING,
            multi_thread=False,
            quadratic=True,

        )
    if track_emg:
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_CONTROL,
            weight=weights["track_emg"],
            target=muscles_target,
            index=muscle_track_idx,
            key="muscles",
            multi_thread=False,
        quadratic=True,

        )
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            weight=weights["min_tracked_activation"],
            index=muscle_track_idx,
            key="muscles",
            multi_thread=False,
        quadratic=True,

        )

    if use_torque:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weights["min_torque"], key="tau", multi_thread=False,
            quadratic=True,
        )

    kin_funct = ObjectiveFcn.Lagrange.TRACK_STATE if kin_data_to_track == "q" else ObjectiveFcn.Lagrange.TRACK_MARKERS
    if kin_data_to_track == "markers":
        objectives.add(
            kin_funct,
            weight=weights["track_kin"],
            target=kin_target,
            node=Node.ALL,
            multi_thread=False,
            quadratic=True,
        )
    elif kin_data_to_track == "q":
        objectives.add(kin_funct, weight=weights["track_kin"], target=kin_target, key="q",
                       node=Node.ALL,
                       multi_thread=False)
    return objectives


def custom_muscles_driven(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    nlp: NonLinearProgram,
    with_residual_torque: bool = True,
    with_f_ext: bool = False,
    external_forces_object = None,
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
    """

    DynamicsFunctions.apply_parameters(nlp)
    q = nlp.get_var_from_states_or_controls("q", states, controls)
    qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
    residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if with_residual_torque else None
    mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

    # for i in range(nlp.model.nb_q):
    #     if i > 4 and i != nlp.model.nb_q - 1:
    #         residual_tau[i] = MX(0)

    tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

    if with_f_ext:
        f_ext = nlp.get_var_from_states_or_controls("f_ext", states, controls)
        #from casadi import cross
        B = [0, 0, 0, 1]
        all_jcs = nlp.model.model.allGlobalJCS(q)
        RT = all_jcs[-1].to_mx()
        # A = RT @ A
        B = RT @ B
        vecteur_OB = B[:3]
        f_ext[:3] = f_ext[:3] + cross(vecteur_OB, f_ext[3:6])
        ext = nlp.model.model.externalForceSet()
        ext.add("hand_left", f_ext)
        ddq = nlp.model.model.ForwardDynamics(q, qdot, tau, ext).to_mx()
    else:
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, False)
    dxdt = MX(nlp.states.shape, ddq.shape[1])
    dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
    dxdt[nlp.states["qdot"].index, :] = ddq

    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def custom_configure(ocp: OptimalControlProgram,
                     nlp: NonLinearProgram,
                     with_residual_torque: bool = True,
                     with_f_ext: bool = False,
                     external_forces_object = None
                     ):
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
    if with_f_ext:
        ConfigureProblem.configure_new_variable("f_ext",
                                                ["mx", "my", "mz", "fx", "fy", "fz"],
                                                ocp,
                                                nlp,
                                                as_states=False,
                                                as_controls=True)
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp,
                                                 nlp,
                                                 custom_muscles_driven,
                                                 with_residual_torque=with_residual_torque,
                                                 with_f_ext=with_f_ext,
                                                 external_forces_object=external_forces_object
                                                 )


def prepare_problem(
    model_path: str,
    objectives: ObjectiveList,
    window_len: int,
    window_duration: float,
    x0: np.ndarray,
    constraints: ConstraintList = None,
    f_ext_object = None,
    f_ext_0 = None,
    use_torque: bool = False,
    with_f_ext: bool = False,
    f_ext_as_constraints: bool = False,
    nb_threads: int = 8,
    solver_options: dict = None,
    use_acados: bool = False,
):
    """
    Prepare the ocp problem and the solver to use

    parameters
    -----------
    model_path : str
        Path to the model
    objectives : ObjectiveList
        List of objectives
    constraints : ConstraintList
        List of constraints
    window_len : int
        Length of the window
    window_duration : float
        Duration of the window
    x0 : np.ndarray
        Initial state
    u0 : np.ndarray
        Initial control
    use_torque : bool
        Use torque as control
    nb_threads : int
        Number of threads to use
    solver_options : dict
        Solver options
    use_acados : bool
        Use acados solver

    Returns
    -------
    The problem and the solver.
    """
    biorbd_model = BiorbdModel(model_path)
    nbGT = biorbd_model.nb_tau if use_torque else 0
    tau_min, tau_max, tau_init = -10000, 10000, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.1
    if with_f_ext and f_ext_as_constraints:
        f_ext_min, f_ext_max, f_ext_init = f_ext_0, f_ext_0, f_ext_0
    elif with_f_ext and f_ext_as_constraints is False:
        pass
    f_ext_min, f_ext_max, f_ext_init = -50, 50, f_ext_0

    # Dynamics
    dynamics = DynamicsList()
    # dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True)
    dynamics.add(
        custom_configure,
        dynamic_function=custom_muscles_driven,
        with_f_ext=with_f_ext,
        external_forces_object=f_ext_object,
        with_residual_torque=True,
        # expand=False,
    )

    # State path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = biorbd_model.bounds_from_ranges("q")
    x_bounds["qdot"] = biorbd_model.bounds_from_ranges("qdot")
    # x_bounds["q"].min[:, 0] = x0[: biorbd_model.nb_q, 0]
    # x_bounds["q"].max[:, 0] = x0[: biorbd_model.nb_q, 0]
    # x_bounds["qdot"].min[:, 0] = x0[biorbd_model.nb_q:, 0]
    # x_bounds["qdot"].max[:, 0] = x0[biorbd_model.nb_q:, 0]

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=np.ones((nbGT)) * tau_min, max_bound=np.ones((nbGT)) * tau_max,
                 interpolation=InterpolationType.CONSTANT)
    u_bounds.add("muscles", min_bound=np.zeros((biorbd_model.nb_muscles)), max_bound=np.ones((biorbd_model.nb_muscles)),
                 interpolation=InterpolationType.CONSTANT)
    if with_f_ext:
        u_bounds.add("f_ext", min_bound=f_ext_min, max_bound=f_ext_max, interpolation=InterpolationType.CONSTANT)
        # u_bounds["f_ext"].min[:, 0] = f_ext_0[:, 0]
        # u_bounds["f_ext"].max[:, 0] = f_ext_0[:, 0]
        # u_bounds["f_ext"].min[:, -1] = f_ext_0[:, -1]
        # u_bounds["f_ext"].max[:, -1] = f_ext_0[:, -1]

    # Initial guesses
    if x0.shape[0] != biorbd_model.nb_q * 2:
        x0 = np.concatenate((x0[:, : window_len + 1], np.zeros((x0.shape[0], window_len + 1))))
    else:
        x0 = x0[:, : window_len + 1]

    x_init = InitialGuessList()
    u_init = InitialGuessList()
    x_init.add("q", x0[:biorbd_model.nb_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", x0[biorbd_model.nb_q:, :], interpolation=InterpolationType.EACH_FRAME)

    u_init.add("tau", np.ones((nbGT)) * tau_init, interpolation=InterpolationType.CONSTANT)
    u_init.add("muscles", np.ones((biorbd_model.nb_muscles)) * muscle_init,
               interpolation=InterpolationType.CONSTANT)

    if with_f_ext:
        u_init.add("f_ext",  f_ext_init, interpolation=InterpolationType.EACH_FRAME)
    #objectives = ObjectiveList()
    problem = CustomMhe(
        bio_model=biorbd_model,
        dynamics=dynamics,
        window_len=window_len,
        window_duration=window_duration,
        common_objective_functions=objectives,
        constraints=constraints,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=nb_threads,
        use_sx=use_acados,
    )
    if use_acados:
        solver_tmp = Solver.ACADOS()
        solver_tmp.set_integrator_type("IRK")
        #solver.set_integrator_type("ERK")
        solver_tmp.set_qp_solver("PARTIAL_CONDENSING_OSQP")
        #solver.set_qp_solver("PARTIAL_CONDENSING_HPIPM")
        solver_tmp.set_nlp_solver_type("SQP_RTI")
        solver_tmp.set_print_level(0)
        for key in solver_options.keys():
            solver_tmp.set_option_unsafe(val=solver_options[key], name=key)

    else:
        solver_tmp = Solver.IPOPT()
        solver_tmp.set_hessian_approximation("exact")
        solver_tmp.set_linear_solver("ma57")
        solver_tmp.set_tol(1e-3)
        solver_tmp.set_print_level(5)
        solver_tmp.set_maximum_iterations(1000)
        for key in solver_options.keys():
            solver_tmp.set_option_unsafe(val=solver_options[key], name=key)
    return problem, solver_tmp


def configure_weights():
    """
    Configure the weights for the objective functions

    Returns
    -------
    weights : dict
        Dictionary of weights
    """
    # weights = {
    #     "min_dq": 1000,
    #     "min_q": 10,
    #     "min_torque": 100,
    #     "min_activation": 100,
    #     "min_tracked_activation": 1,
    #     "track_emg": 1000,
    #     "previous_q": 10,
    #     "previous_q_dot": 10,
    #     "track_kin": 100000000,
    #     "f_ext": 1000000,
    # }

    weights = {
    "min_dq": 10,
    "min_q": 1,
    "min_torque": 500,
    "min_activation": 100,
    "min_tracked_activation": 10,
    "track_emg": 1000000,
    "previous_q": 100,
    "previous_q_dot": 100,
    "track_kin": 100000000,
    "f_ext": 10000000,
    }

    # tmhe 0.09
    # weights = {
    #     "track_markers": 1000000000000000,
    #     "track_q": 100000000000000,
    #     "min_control": 120000000000,
    #     "min_dq": 10000000,
    #     "min_q": 1,
    #     "min_torque": 1000,
    #     # "track_emg": 10000000000000,
    #     "track_emg": 3800000000000,
    #     "min_activation": 10,
    #
    # }
    return weights


def get_target(
    mhe,
    x_ref: np.ndarray,
    markers_ref: np.ndarray,
    muscles_ref: np.ndarray,
    f_ext_ref: np.ndarray,
    ns_mhe: int,
    slide_size: int,
    track_emg: bool,
    kin_data_to_track: str,
    model: BiorbdModel,
    sol,
):
    """
    Get the target for the next MHE problem and the objective functions index.

    Parameters
    ----------
    mhe : CustomMhe
        The MHE problem
    t : float
        The current time
    x_ref : np.ndarray
        The reference state
    markers_ref : np.ndarray
        The reference markers
    muscles_ref : np.ndarray
        The reference muscles
    ns_mhe : int
        The number of node of the MHE problem
    slide_size : int
        The size of the sliding window
    track_emg : bool
        Whether to track EMG
    kin_data_to_track : str
        The kin_data to track
    model : biorbd.Model
        The model
    offline : bool
        Whether to use offline data

    Returns
    -------
    Dictionary of targets (values and objective functions index)
    """
    nbMT, nbQ, nbF = model.nb_muscles, model.nb_q, 6
    muscles_ref = muscles_ref if track_emg is True else np.zeros((nbMT, ns_mhe))
    q_target_idx, markers_target_idx, muscles_target_idx, f_ext_target_idx = [], [], [], []
    if sol:
        previous_sol = np.zeros((nbQ * 2, ns_mhe + 1))
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        previous_sol[:nbQ, :] = np.concatenate((states["q"][:, slide_size:],
                                                np.repeat(states["q"][:, -1:], slide_size,
                                                          axis=1)),
                                               axis=1)
        previous_sol[nbQ: nbQ * 2, :] = np.concatenate((states["qdot"][:, slide_size:],
                                                        np.repeat(states["qdot"][:, -1:], slide_size,
                                                                  axis=1)),
                                                       axis=1)
    else:
        previous_sol = np.concatenate((x_ref, np.zeros((x_ref.shape[0], ns_mhe + 1))))

    # Find objective function idx for targets
    for i in range(len(mhe.nlp[0].J)):
        if mhe.nlp[0].J[i].name == "MINIMIZE_CONTROL" and mhe.nlp[0].J[i].target is not None:
            if mhe.nlp[0].J[i].params["key"] == "muscles":
                muscles_target_idx.append(i)
            if mhe.nlp[0].J[i].params["key"] == "f_ext":
                f_ext_target_idx.append(i)
        elif mhe.nlp[0].J[i].name == "MINIMIZE_MARKERS" and mhe.nlp[0].J[i].target is not None:
            markers_target_idx.append(i)
        elif mhe.nlp[0].J[i].name == "MINIMIZE_STATE" and mhe.nlp[0].J[i].target is not None:
            q_target_idx.append(i)
    kin_target_idx = q_target_idx if kin_data_to_track == "q" else markers_target_idx

    # Define target
    if kin_data_to_track == "q":
        kin_target = x_ref[:nbQ, :].copy()
    else:
        kin_target = markers_ref

    target = {
        "kin_target": [kin_target_idx[0], kin_target]}

    if len(q_target_idx) != 0 and kin_data_to_track == "markers":
        target["previous_q"] = [q_target_idx[0], previous_sol[:nbQ, :]]
        target["previous_q_dot"] = [q_target_idx[1], previous_sol[nbQ: nbQ * 2, :]]

    if track_emg:
        target["muscle_target"] = [muscles_target_idx[0], muscles_ref]
    mhe.muscles_ref = muscles_ref
    mhe.muscles_target = None if not track_emg else muscles_ref

    if len(f_ext_target_idx) != 0:
        target["f_ext_target"] = [f_ext_target_idx[0], f_ext_ref]
    mhe.f_ext_ref = f_ext_ref
    mhe.f_ext_target = None if len(f_ext_target_idx) == 0 else f_ext_ref

    mhe.x_ref = x_ref.copy()
    if kin_data_to_track == "q":
        mhe.kin_target = target["kin_target"][1]
    else:
        if isinstance(target["kin_target"][1], list):
            mhe.kin_target = np.concatenate((target["kin_target"][1][0], target["kin_target"][1][1]), axis=1)
        else:
            mhe.kin_target = target["kin_target"][1]

    for key in target.keys():
        if isinstance(target[key][1], list):
            for i in range(len(target[key][1])):
                mhe.update_objectives_target(target=target[key][1][i], list_index=target[key][0][i])
        else:
            mhe.update_objectives_target(target=target[key][1], list_index=target[key][0])
    return mhe


def update_mhe(mhe, t: int, sol: bioptim.Solution, ei, initial_time: float):
    """
    Update the MHE problem with the current data.

    Parameters
    ----------
    mhe : CustomMhe
        The MHE problem
    t : int
        The current time
    sol : bioptim.Solution
        The solution of the previous problem
    ei : instance of the estimator class
        The estimator instance
    initial_time : float
        The initial time
    offline_data : bool
        Whether to use offline data

    Returns
    -------
    if online : True
    else : True if there are still target available, False otherwise
    """
    tic = time()
    slide_size = ei.slide_size
    ns_mhe = ei.ns_mhe

    x_ref_to_save = [] if mhe.x_ref is None else mhe.x_ref.copy()
    muscles_target_to_save = mhe.muscles_ref if mhe.muscles_ref is not None else []
    kin_target_to_save = mhe.kin_target if mhe.kin_target is not None else []
    f_ext_ref_to_save = mhe.f_ext_ref if mhe.f_ext_ref is not None else []
    if t == 0:
        x_init = {}
        for key in mhe.nlp[0].x_init.keys():
            x_init[key] = np.array(mhe.nlp[0].x_init[key].init[:, :]).copy()
        u_init = {}
        for key in mhe.nlp[0].u_init.keys():
            u_init[key] = np.array(mhe.nlp[0].u_init[key].init[:, :]).copy()
        mhe.x_init = x_init
        mhe.u_init = u_init

    # n_before = ei.n_before_interpolate
    # x_ref, markers_target, muscles_target, f_ext_target = ei.offline_data
    # # advance window
    # markers_target = markers_target[:, :, slide_size * t: (n_before + 1 + slide_size * t)]
    # x_ref = x_ref[:, slide_size * t: (n_before + 1 + slide_size * t)].copy()
    # muscles_target = muscles_target[:, slide_size * t: (n_before + 1 + slide_size * t)]
    # f_ext_target = f_ext_target[:, slide_size * t: (n_before + 1 + slide_size * t)]
    #
    # x_ref, markers_ref, muscles_target, f_ext_ref = interpolate_data(
    #     ei.interpol_factor, x_ref, muscles_target, markers_target, f_ext_target=f_ext_target
    # )
    #
    # muscles_target = muscles_target[:, : ns_mhe]
    # f_ext_ref = f_ext_ref[:, : ns_mhe]
    # markers_ref = markers_ref[:, :, : ns_mhe + 1]
    # x_ref = x_ref[:, : ns_mhe + 1]
    markers_ref = ei.markers_target[:, :, slide_size * t: slide_size * t + ns_mhe + 1].copy()
    x_ref = ei.x_ref[:, slide_size * t: slide_size * t + ns_mhe + 1].copy()
    muscles_ref = ei.muscles_target[:, slide_size * t: slide_size * t + ns_mhe + 1][..., :-1].copy()
    f_ext_ref = ei.f_ext_target[:, slide_size * t: slide_size * t + ns_mhe + 1][..., :-1].copy()
    tic = time()
    mhe = get_target(
        mhe=mhe,
        x_ref=x_ref,
        markers_ref=markers_ref,
        muscles_ref=muscles_ref,
        f_ext_ref=f_ext_ref,
        ns_mhe=ei.ns_mhe,
        slide_size=ei.slide_size,
        track_emg=ei.track_emg,
        kin_data_to_track=ei.kin_data_to_track,
        model=ei.biorbd_model,
        sol=sol,
    )

    stat = -1
    if t > 0:
        stat = sol.status  # if sol.status != 0 else -1
        tmp_slide_size = ei.slide_size
        ei.slide_size = 1
        q_est, dq_est, a_est, u_est, force_est, f_ext, tau = compute_force(
            sol,
            ei.get_force,
            ei.nbMT,
            frame_to_save=ei.frame_to_save,
            slide_size=ei.slide_size,
            save_all_frame=ei.save_all_frame,
        )
        time_to_get_data = time() - tic
        time_to_solve = sol.real_time_to_optimize
        # time_tot = time_to_solve + time_to_get_data
        time_tot = time_to_solve

        if ei.save_results:
            dic_to_save = {
                "q_est": q_est,
                "dq_est": dq_est,
                "u_est": u_est,
                "muscle_force": force_est,
                "tau_est": tau,
                "iter": t,
                "time": time() - initial_time,
                "stat": stat,
                "x_ref": x_ref_to_save,
                "x_init": mhe.x_init,
                "u_init": mhe.u_init,
                "muscles_target": muscles_target_to_save,
                "f_ext": f_ext,
                "f_ext_ref": f_ext_ref_to_save,
                "n_mhe": ei.ns_mhe,
                "sol_freq": 1 / time_tot,
                "sleep_time": (1 / ei.exp_freq) - time_tot,
                "kin_target": kin_target_to_save,
                "exp_freq": ei.exp_freq,
                "frame_to_export": ei.frame_to_save,
                "save_all_frame": ei.save_all_frame,
                "slide_size": ei.slide_size,
                "muscle_track_idx": ei.muscle_track_idx,
            }
            data_path = ei.result_dir + os.sep + ei.result_file_name
            save(dic_to_save, data_path, add_data=True)
            if ei.print_lvl == 1:
                print(
                    "solver status : ", stat, "\n"
                    f"Solve Frequency : {1 / time_tot} \n"
                    f"Expected Frequency : {ei.exp_freq}\n"
                    #f"time to sleep: {(1 / ei.exp_freq) - time_tot}\n"
                    #f"time to get data = {time_to_get_data}"
                )
            x_init = {}
            for key in mhe.nlp[0].x_init.keys():
                x_init[key] = mhe.nlp[0].x_init[key].init[:, :]
            u_init = {}
            for key in mhe.nlp[0].u_init.keys():
                u_init[key] = mhe.nlp[0].u_init[key].init[:, :]
            mhe.x_init = x_init
            mhe.u_init = u_init

        current_time = time() - tic
        time_tot = time_to_solve + current_time
        if 1 / time_tot > ei.exp_freq:
            sleep((1 / ei.exp_freq) - time_tot)
        ei.slide_size = tmp_slide_size

    if t == 810:
        # plt.figure("n")
        # plt.plot()
        # plt.show()
        return False
    else:
        return True
    # try:
    #     if mhe.kin_target.shape[2] > ei.ns_mhe:
    #         return True
    # except:
    #     if mhe.kin_target.shape[1] > ei.ns_mhe:
    #         return True
    # else:
    #     return False


class CustomMhe(MovingHorizonEstimator):
    """
    Class for the custom MHE.
    """

    def __init__(self, **kwargs):
        self.x_ref = None
        self.muscles_ref = None
        self.f_ext = None
        self.with_f_ext = False
        self.f_ext_ref = None
        self.f_ext_as_constraints = False
        self.kin_target = None
        self.slide_size = 1
        self.f_x, self.f_u, = (
            None,
            None,
        )
        self.kalman = None
        super(CustomMhe, self).__init__(**kwargs)

    def advance_window_initial_guess_states(self, sol, **advance_options):
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        for key in states.keys():
            if self.nlp[0].x_init[key].type != InterpolationType.EACH_FRAME:
                # Override the previous x_init
                self.nlp[0].x_init.add(
                    key, np.ndarray(states[key].shape), interpolation=InterpolationType.EACH_FRAME, phase=0
                )
                self.nlp[0].x_init[key].check_and_adjust_dimensions(len(self.nlp[0].states[key]), self.nlp[0].ns)

            self.nlp[0].x_init[key].init[:, :] = np.concatenate(
                (states[key][:, self.slide_size:],
                 np.repeat(states[key][:, -1][:, np.newaxis], self.slide_size, axis=1),)
                , axis=1
            )
        return True

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        for key in self.nlp[0].u_init.keys():
            self.nlp[0].controls.node_index = 0
            if self.nlp[0].u_init[key].type != InterpolationType.EACH_FRAME:
                # Override the previous u_init
                self.nlp[0].u_init.add(
                    key,
                    np.ndarray((controls[key].shape[0], self.nlp[0].n_controls_nodes)),
                    interpolation=InterpolationType.EACH_FRAME,
                    phase=0,
                )
                self.nlp[0].u_init[key].check_and_adjust_dimensions(
                    len(self.nlp[0].controls[key]), self.nlp[0].n_controls_nodes - 1
                )

            self.nlp[0].u_init[key].init[:, :] = np.concatenate(
                (controls[key][:, self.slide_size:],
                 np.repeat(controls[key][:, -1][:, np.newaxis], self.slide_size, axis=1),)
                 , axis=1
            )
        return True

    def advance_window_bounds_states(self, sol, **advance_options):
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        for key in states.keys():
            self.nlp[0].x_bounds[key].min[:, 0] = states[key][:, self.slide_size]
            self.nlp[0].x_bounds[key].max[:, 0] = states[key][:, self.slide_size]
        return True

    # def advance_window_bounds_controls(self, sol, **advance_options):
    #     if self.f_ext_as_constraints:
    #         self.nlp[0].u_bounds.min[-6:, 0] = self.f_ext[:, 0]
    #         self.nlp[0].u_bounds.max[-6:, 0] = self.f_ext[:, 0]
    #         self.nlp[0].u_bounds.min[-6:, 1] = self.f_ext[:, 1:-1]
    #         self.nlp[0].u_bounds.max[-6:, 1] = self.f_ext[:, 1:-1]
    #         self.nlp[0].u_bounds.min[-6:, -1] = self.f_ext[:, -1]
    #         self.nlp[0].u_bounds.max[-6:, -1] = self.f_ext[:, -1]
    #         return True
    #     else:
    #         return False

    # def export_data(self, sol) -> tuple:
    #     return (
    #         sol.decision_states(to_merge=SolutionMerge.NODES)[:, self.frame_to_export: self.frame_to_export + 1],
    #         sol.decision_controls(to_merge=SolutionMerge.NODES)[:, self.frame_to_export: self.frame_to_export + 1],
    #     )

    # def _initialize_solution(self, states: list, controls: list):
    #     _states = InitialGuess(np.concatenate(states, axis=1), interpolation=InterpolationType.EACH_FRAME)
    #     _controls = InitialGuess(np.concatenate(controls, axis=1), interpolation=InterpolationType.EACH_FRAME)
    #     solution_ocp = OptimalControlProgram(
    #         bio_model=self.original_values["bio_model"][0],
    #         dynamics=self.original_values["dynamics"][0],
    #         n_shooting=(self.total_optimization_run * 1) - 1,
    #         phase_time=self.total_optimization_run * self.nlp[0].dt,
    #         skip_continuity=True,
    #     )
    #     return Solution(solution_ocp, [_states, _controls])

    # def _initialize_solution(self, dt: float, states: list, controls: list):
    #     x_init = InitialGuessList()
    #     for key in self.nlp[0].states.keys():
    #         x_init.add(
    #             key,
    #             np.concatenate([state[key] for state in states], axis=1),
    #             interpolation=InterpolationType.EACH_FRAME,
    #             phase=0,
    #         )
    #
    #     u_init = InitialGuessList()
    #     for key in self.nlp[0].controls.keys():
    #         controls_tp = np.concatenate([control[key] for control in controls], axis=1)
    #         u_init.add(key, controls_tp, interpolation=InterpolationType.EACH_FRAME, phase=0)
    #
    #     model_serialized = self.nlp[0].model.serialize()
    #     model_class = model_serialized[0]
    #     model_initializer = model_serialized[1]
    #
    #     solution_ocp = OptimalControlProgram(
    #         bio_model=model_class(**model_initializer),
    #         dynamics=self.nlp[0].dynamics_type,
    #         n_shooting=self.total_optimization_run,
    #         phase_time=self.total_optimization_run * dt,
    #         x_init=x_init,
    #         u_init=u_init,
    #         use_sx=self.cx == SX,
    #     )
    #     a_init = InitialGuessList()
    #     p_init = InitialGuessList()
    #     return Solution.from_initial_guess(solution_ocp, [np.array([dt]), x_init, u_init, p_init, a_init])
    def solve(
        self,
        update_function,
        solver =  None,
        warm_start = None,
        solver_first_iter = None,
        export_options: dict = None,
        max_consecutive_failing=np.inf,
        update_function_extra_params: dict = None,
        get_all_iterations: bool = False,
        **advance_options,
    ) :
        """
        Solve MHE program. The program runs until 'update_function' returns False. This function can be used to
        modify the objective set, for instance. The warm_start_function can be provided by the user. Otherwise, the
        initial guess is the solution where the first frame is dropped and the last frame is duplicated. Moreover,
        the bounds at first frame is set to the new first frame of the initial guess

        Parameters
        ----------
        update_function: Callable
            A function with the signature: update_function(mhe, current_time_index, previous_solution), where the
            mhe is the current program, current_time_index starts at 0 and increments after each solve and
            previous_solution is None the first call and then is the Solution structure for the last solving of the MHE.
            The function 'update_function' is called before each solve. If it returns true, the next frame is solve.
            Otherwise, it finishes the MHE and the solution is returned. The `update_function` callback can also
            be used to modify the program (usually the targets of some objective functions) and initial condition and
            bounds.
        solver: Solver
            The Solver to use (default being ACADOS)
        solver: Solver
            The Solver to use for the first iteration (must be the same as solver, but more options can be changed)
        warm_start: Solution
            A Solution to initiate the first iteration from
        export_options: dict
            Any options related to the saving of the data at each iteration
        max_consecutive_failing: int
            The number of consecutive failing before stopping the nmpc. Default is infinite
        update_function_extra_params: dict
            Any parameters to pass to the update function
        get_all_iterations: bool
            If an extra output value that includes all the individual solution should be returned
        advance_options: Any
            The extra options to pass to the advancing methods

        Returns
        -------
        The solution of the MHE
        """

        if len(self.nlp) != 1:
            raise NotImplementedError("MHE is only available for 1 phase program")

        sol = None
        states = []
        controls = []

        solver_all_iter = Solver.ACADOS() if solver is None else solver
        if solver_first_iter is None and solver is not None:
            # If not first iter was sent, the all iter becomes the first and is not updated afterward
            solver_first_iter = solver_all_iter
            solver_all_iter = None
        solver_current = solver_first_iter

        self._initialize_frame_to_export(export_options)

        total_time = 0
        all_solutions = []
        split_solutions = []
        consecutive_failing = 0
        update_function_extra_params = {} if update_function_extra_params is None else update_function_extra_params

        self.total_optimization_run = 0
        while (
            update_function(self, self.total_optimization_run, sol, **update_function_extra_params)
            and consecutive_failing < max_consecutive_failing
        ):
            sol = super(RecedingHorizonOptimization, self).solve(
                solver=solver_current,
                warm_start=warm_start,
            )
            self.sol = sol
            consecutive_failing = 0 if sol.status == 0 else consecutive_failing + 1

            # Set the option for the next iteration
            if self.total_optimization_run == 0:
                # Update the solver if first and the rest are different
                if solver_all_iter:
                    solver_current = solver_all_iter
                    if solver_current.type == SolverType.ACADOS and solver_current.only_first_options_has_changed:
                        raise RuntimeError(
                            f"Some options has been changed for the second iteration of acados.\n"
                            f"Only {solver_current.get_tolerance_keys()} can be modified."
                        )
                if solver_current.type == SolverType.IPOPT:
                    solver_current.show_online_optim = False
            warm_start = None

            total_time += sol.real_time_to_optimize
# Reset timer to skip the compiling time (so skip the first call to solve)

            # Solve and save the current window of interest
            _states, _controls = self.export_data(sol)
            states.append(_states)
            controls.append(_controls)
            # Solve and save the full window of the OCP
            if get_all_iterations:
                all_solutions.append(sol)
            # Update the initial frame bounds and initial guess
            self.advance_window(sol, **advance_options)

            self.total_optimization_run += 1

        states.append({key: sol.decision_states()[key][-1] for key in sol.decision_states().keys()})

        # Prepare the modified ocp that fits the solution dimension
        dt = sol.t_span()[0][-1]
        # final_sol = states, controls
        return