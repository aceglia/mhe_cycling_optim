"""
This code provide every function needed to solve the OCP problem.
"""
import bioptim
from .utils import *
from time import time, sleep
import biorbd_casadi as biorbd
from biosiglive.file_io.save_and_load import load
import numpy as np
import datetime
import scipy.io as sio
from casadi import MX, Function, horzcat, vertcat, mtimes, cross
from bioptim import (
    MovingHorizonEstimator,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    InitialGuessList,
    BoundsList,
    InterpolationType,
    Solver,
    Node,
    OptimalControlProgram,
    Solution,
    DynamicsFunctions,
    DynamicsEvaluation,
    BiorbdModel,
    NonLinearProgram,
    ConfigureProblem,
    ConstraintList,
    ConstraintFcn,
)


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
    muscle_min_idx = []
    for i in range(biorbd_model.nb_muscles):
        if i not in muscle_track_idx:
            muscle_min_idx.append(i)

    objectives = ObjectiveList()
    if track_emg:
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_CONTROL,
            weight=weights["track_emg"],
            target=muscles_target,
            index=muscle_track_idx,
            key="muscles",
            multi_thread=False,
        )
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            weight=10,
            # tmhe = 0.09 :
            # weight=100000000,
            index=muscle_track_idx,
            key="muscles",
            multi_thread=False,
        )

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weights["min_control"], key="muscles", multi_thread=False
    )

    if use_torque:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weights["min_torque"], key="tau", multi_thread=False
        )
    kin_funct = ObjectiveFcn.Lagrange.TRACK_STATE if kin_data_to_track == "q" else ObjectiveFcn.Lagrange.TRACK_MARKERS
    kin_weight = weights["track_markers"] if kin_data_to_track == "markers" else weights["track_q"]
    if kin_data_to_track == "markers":
        objectives.add(
            kin_funct,
            weight=kin_weight,
            target=kin_target,
            node=Node.ALL,
            multi_thread=False,
        )

    elif kin_data_to_track == "q":
        objectives.add(kin_funct, weight=kin_weight, target=kin_target, key="q", node=Node.ALL, multi_thread=False)

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_dq"],
        index=np.array(range(biorbd_model.nb_q)),
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
    )

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_dq"],
        index=np.array(range(biorbd_model.nb_q)),
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
    )
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=100000000,
        target=previous_q[:, :],
        key="q",
        node=Node.ALL,
        multi_thread=False,
    )
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=100000,
        target=previous_qdot[:, :],
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
    )
    if with_f_ext and f_ext_as_constraints is False:
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_CONTROL,
            weight=1,
            target=f_ext_target,
            key="f_ext",
            node=Node.ALL_SHOOTING,
            multi_thread=False,
        )
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
    residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls)  if with_residual_torque else None

    mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)

    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

    # for i in range(nlp.model.nb_q):
    #     if i > 4 and i != nlp.model.nb_q - 1:
    #         residual_tau[i] = MX(0)

    tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    if with_f_ext:
        f_ext = nlp.get_var_from_states_or_controls("f_ext", states, controls)
        B = [0, 0, 0, 1]
        f_ext_mat = MX(6, 1)
        all_jcs = nlp.model.model.allGlobalJCS(q)
        RT = all_jcs[-1].to_mx()
        B = mtimes(RT, B)
        vecteur_OB = B[:3]
        f_ext_mat[:3] = f_ext[:3] + cross(vecteur_OB, f_ext[3:6])
        f_ext_mat[3:] = f_ext[3:]
        external_forces_object.add("radius_left_pro_sup_left", f_ext_mat)
        ddq = nlp.model.model.ForwardDynamics(q, qdot, tau, external_forces_object).to_mx()
    else:
        # nlp.external_forces = np.zeros((6,16))
        # ddq = nlp.model.constrained_forward_dynamics(q, qdot, tau, np.zeros((6, 16)))
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, False)
    dxdt = MX(nlp.states.shape, ddq.shape[1])
    dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
    dxdt[nlp.states["qdot"].index, :] = ddq

    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, with_residual_torque: bool = True,
    with_f_ext: bool = False,
    external_forces_object = None,):
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
    if with_f_ext:
        ConfigureProblem.configure_new_variable("f_ext", [f"f_ext_{i}" for i in range(6)],
                                                ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_muscles_driven,
                                                 with_residual_torque=with_residual_torque,
    with_f_ext= with_f_ext,
    external_forces_object=external_forces_object,)


def prepare_problem(
    model_path: str,
    objectives: ObjectiveList,
    constraints: ConstraintList,
    window_len: int,
    window_duration: float,
    x0: np.ndarray,
    u0: np.ndarray = None,
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
    tau_min, tau_max, tau_init = -30, 30, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.1
    if with_f_ext and f_ext_as_constraints:
        f_ext_min, f_ext_max, f_ext_init = f_ext_0, f_ext_0, f_ext_0
    elif with_f_ext and f_ext_as_constraints is False:
        f_ext_min, f_ext_max, f_ext_init = [-500, -500, -500, -500, -500, -500], [500, 500, 500, 500, 500, 500], f_ext_0


    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        # DynamicsFcn.MUSCLE_DRIVEN,
        custom_configure,
        dynamic_function=custom_muscles_driven,
        # dynamic_function=custom_dynamic,
        # with_excitations=False,
        # with_torque=use_torque,
        with_f_ext=with_f_ext,
        external_forces_object=f_ext_object,
        with_residual_torque=True,
        # expand=False,
    )

    if x0.shape[0] != biorbd_model.nb_q * 2:
        x_0 = np.concatenate((x0[:, : window_len + 1], np.zeros((x0.shape[0], window_len + 1))))
    else:
        x_0 = x0[:, : window_len + 1]

    # State path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = biorbd_model.bounds_from_ranges("q")
    x_bounds["qdot"] = biorbd_model.bounds_from_ranges("qdot")
    # x_bounds[0].min[: biorbd_model.nb_q, 0] = [i - 0.1 * i for i in x_0[: biorbd_model.nb_q, 0]]
    # x_bounds[0].max[: biorbd_model.nb_q, 0] = [i + 0.1 * i for i in x_0[: biorbd_model.nb_q, 0]]
    # x_bounds[0].min[biorbd_model.nb_q: biorbd_model.nb_q * 2, 0] = [
    #     i - 0.1 * i for i in x_0[biorbd_model.nb_q:, 0]
    # ]
    # x_bounds[0].max[biorbd_model.nb_q: biorbd_model.nb_q * 2, 0] = [
    #     i + 0.1 * i for i in x_0[biorbd_model.nb_q:, 0]
    # ]
    # x_bounds[0].min[biorbd_model.nb_q: biorbd_model.nb_q * 2, [1, -1]] = [[-5] * 2] * biorbd_model.nb_q
    # x_bounds[0].max[biorbd_model.nb_q: biorbd_model.nb_q * 2, [1, -1]] = [[5] * 2] * biorbd_model.nb_q
    if with_f_ext and f_ext_as_constraints:
        u_bounds = BoundsList()
        u_bounds.add("tau", min_bound=tau_min, max_bound=tau_max, interpolation=InterpolationType.CONSTANT)
        u_bounds.add("muscles", min_bound=muscle_min, max_bound=muscle_max, interpolation=InterpolationType.CONSTANT)
        u_bounds.add("f_ext", min_bound=f_ext_0[:, : window_len],
                     max_bound=f_ext_0[:, : window_len], interpolation=InterpolationType.EACH_FRAME)
    else:
        u_bounds = BoundsList()
        u_bounds.add("tau", min_bound=tau_min, max_bound=tau_max, interpolation=InterpolationType.CONSTANT)
        u_bounds.add("muscles", min_bound=muscle_min, max_bound=muscle_max, interpolation=InterpolationType.CONSTANT)
    # Initial guesses
    if x0.shape[0] != biorbd_model.nb_q * 2:
        x0 = np.concatenate((x0[:, : window_len + 1], np.zeros((x0.shape[0], window_len + 1))))
    else:
        x0 = x0[:, : window_len + 1]
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    x_init.add("q", x0[:biorbd_model.nb_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", x0[biorbd_model.nb_q:, :], interpolation=InterpolationType.EACH_FRAME)
    if with_f_ext:
        if u0 is None:
            u0 = np.concatenate((np.tile(np.array([tau_init] * nbGT + [muscle_init] * biorbd_model.nb_muscles),
                                         (window_len, 1)).T, f_ext_0), axis=0)
        u_init.add("tau", u0[:nbGT, :], interpolation=InterpolationType.EACH_FRAME)
        u_init.add("muscles", u0[nbGT:nbGT + biorbd_model.nb_muscles, :], interpolation=InterpolationType.EACH_FRAME)
        if with_f_ext:
            u_init.add("f_ext", u0[nbGT + biorbd_model.nb_muscles:, :], interpolation=InterpolationType.EACH_FRAME)

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
    solver = Solver.ACADOS()
    solver.set_integrator_type("IRK")
    # solver.set_integrator_type("ERK")

    solver.set_qp_solver("PARTIAL_CONDENSING_OSQP")
    # solver.set_qp_solver("PARTIAL_CONDENSING_HPIPM")
    solver.set_nlp_solver_type("SQP_RTI")
    # solver.set_nlp_solver_type("SQP")

    solver.set_print_level(0)
    for key in solver_options.keys():
        solver.set_option_unsafe(val=solver_options[key], name=key)
    return problem, solver


def configure_weights():
    """
    Configure the weights for the objective functions

    Returns
    -------
    weights : dict
        Dictionary of weights
    """
    weights = {
        "track_markers": 10000000000000,
        # "track_q": 100000000000000,
        "min_control": 10000000,
        "min_dq": 10000,
        "min_q": 1,
        "min_torque": 1000000,
        "track_emg": 1000000000,
        "min_activation": 10,
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
    t: float,
    x_ref: np.ndarray,
    markers_ref: np.ndarray,
    muscles_ref: np.ndarray,
    f_ext_ref: np.ndarray,
    ns_mhe: int,
    slide_size: int,
    track_emg: bool,
    kin_data_to_track: str,
    model: BiorbdModel,
    offline: bool,
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
    kin_target_idx = q_target_idx[0] if kin_data_to_track == "q" else markers_target_idx

    # Define target
    muscle_target = muscles_ref[:, slide_size * t: (ns_mhe + 1 + slide_size * t)] if offline else muscles_ref
    f_ext_target = f_ext_ref[:, slide_size*t: (ns_mhe + 1 + slide_size * t)] if offline else f_ext_ref

    if sol:
        previous_sol = np.concatenate(
            (sol.states["all"][:, slide_size:], np.repeat(sol.states["all"][:, -1][:, np.newaxis], slide_size, axis=1)),
            axis=1,
        )
    else:
        previous_sol = np.concatenate((x_ref[:, : ns_mhe + 1], np.zeros((x_ref.shape[0], ns_mhe + 1))))

    if kin_data_to_track == "q":
        kin_target = x_ref[:nbQ, slide_size * t: (ns_mhe + 1 + slide_size * t)] if offline else x_ref
    else:
        kin_target = markers_ref[:3, :, slide_size * t: (ns_mhe + 1 + slide_size * t)]
    target = {
        "kin_target": [kin_target_idx[0], kin_target],
        "previous_q": [q_target_idx[0], previous_sol[:nbQ, :]],
        "previous_q_dot": [q_target_idx[1], previous_sol[nbQ: nbQ * 2, :]],
    }
    if track_emg:
        target["muscle_target"] = [muscles_target_idx[0], muscle_target]
    if len(f_ext_target_idx) != 0:
        target["f_ext_target"] = [f_ext_target_idx[0], f_ext_target[0, :, :].T]
    return target


def update_mhe(mhe, t: int, sol: bioptim.Solution, estimator_instance, initial_time: float, offline_data: bool = None):
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
    estimator_instance : instance of the estimator class
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
    mhe.frame_to_export = estimator_instance.frame_to_save
    # mhe.f_ext_as_constraints = estimator_instance.f_ext_as_constraints
    # target to save
    x_ref_to_save = []
    muscles_target_to_save = []
    vicon_latency = None
    kin_target_to_save = []
    if mhe.x_ref is not None:
        x_ref_to_save = mhe.x_ref
        muscles_target_to_save = mhe.muscles_ref
        kin_target_to_save = mhe.kin_target

    absolute_time_frame = 0
    absolute_delay_tcp = 0
    if estimator_instance.test_offline:
        x_ref, markers_target, muscles_target, f_ext_target = offline_data
        absolute_time_received = datetime.datetime.now()
        absolute_time_received_dic = {
            "day": absolute_time_received.day,
            "hour": absolute_time_received.hour,
            "hour_s": absolute_time_received.hour * 3600,
            "minute": absolute_time_received.minute,
            "minute_s": absolute_time_received.minute * 60,
            "second": absolute_time_received.second,
            "millisecond": int(absolute_time_received.microsecond / 1000),
            "millisecond_s": int(absolute_time_received.microsecond / 1000) * 0.001,
        }
        absolute_time_received_s = 0
        for key in absolute_time_received_dic.keys():
            if key == "second" or key[-1:] == "s":
                absolute_time_received_s = absolute_time_received_s + absolute_time_received_dic[key]
    else:
        data = get_data(
            ip=estimator_instance.server_ip, port=estimator_instance.server_port, message=estimator_instance.message
        )
        x_ref = np.array(data["kalman"])
        markers_target = np.array(data["markers"])
        muscles_target = np.array(data["emg_proc"])
        f_ext_target = np.array(data["f_ext"])
        absolute_time_frame = data["absolute_time_frame"]
        vicon_latency = data["vicon_latency"]
        absolute_time_received = datetime.datetime.now()
        absolute_time_received_dic = {
            "day": absolute_time_received.day,
            "hour": absolute_time_received.hour,
            "hour_s": absolute_time_received.hour * 3600,
            "minute": absolute_time_received.minute,
            "minute_s": absolute_time_received.minute * 60,
            "second": absolute_time_received.second,
            "millisecond": int(absolute_time_received.microsecond / 1000),
            "millisecond_s": int(absolute_time_received.microsecond / 1000) * 0.001,
        }

        absolute_time_frame_s = 0
        absolute_time_received_s = 0
        for key in absolute_time_frame.keys():
            if key == "second" or key[-1:] == "s":
                absolute_time_frame_s = absolute_time_frame_s + absolute_time_frame[key]
                absolute_time_received_s = absolute_time_received_s + absolute_time_received_dic[key]
        absolute_delay_tcp = absolute_time_received_s - absolute_time_frame_s

    x_ref, markers_ref, muscles_target, f_ext_ref = interpolate_data(
        estimator_instance.interpol_factor, x_ref, muscles_target, markers_target, f_ext_target=f_ext_target
    )

    muscles_ref = muscle_mapping(
        muscles_target_tmp=muscles_target,
        mvc_list=estimator_instance.mvc_list,
        muscle_track_idx=estimator_instance.muscle_track_idx,
    )
    target = get_target(
        mhe=mhe,
        t=t,
        x_ref=x_ref,
        markers_ref=markers_ref,
        muscles_ref=muscles_ref,
        f_ext_ref=f_ext_ref,
        ns_mhe=estimator_instance.ns_mhe,
        slide_size=estimator_instance.slide_size,
        track_emg=estimator_instance.track_emg,
        kin_data_to_track=estimator_instance.kin_data_to_track,
        model=estimator_instance.model,
        offline=estimator_instance.test_offline,
        sol=sol,
    )
    mhe.slide_size = estimator_instance.slide_size
    if estimator_instance.test_offline:
        mhe.x_ref = x_ref[
            :, estimator_instance.slide_size * t: (estimator_instance.ns_mhe + 1 + estimator_instance.slide_size * t)
        ]
    else:
        mhe.x_ref = x_ref
    try:
        mhe.muscles_ref = target["muscle_target"][1]
    except:
        mhe.muscles_ref = np.zeros((15, estimator_instance.ns_mhe))
    try:
        mhe.f_ext_ref = target["f_ext_target"][1]
    except:
        mhe.f_ext_ref = np.zeros((6, estimator_instance.ns_mhe))
    if estimator_instance.kin_data_to_track == "q":
        mhe.kin_target = target["kin_target"][1]
    else:
        if isinstance(target["kin_target"][1], list):
            mhe.kin_target = np.concatenate((target["kin_target"][1][0], target["kin_target"][1][1]), axis=1)
        else:
            mhe.kin_target = target["kin_target"][1]
    try:
        mhe.f_ext = target["f_ext_target"][1]
    except:
        pass

    for key in target.keys():
        if isinstance(target[key][1], list):
            for i in range(len(target[key][1])):
                mhe.update_objectives_target(target=target[key][1][i], list_index=target[key][0][i])
        else:
            mhe.update_objectives_target(target=target[key][1], list_index=target[key][0])

    if t != 0:
        stat = t if sol.status != 0 else -1
    else:
        stat = -1

    if t > 0:
        tmp_slide_size = estimator_instance.slide_size
        estimator_instance.slide_size = 1
        q_est, dq_est, a_est, u_est, force_est = compute_force(
            sol,
            estimator_instance.get_force,
            estimator_instance.nbMT,
            frame_to_save=estimator_instance.frame_to_save,
            slide_size=estimator_instance.slide_size,
            save_all_frame=estimator_instance.save_all_frame,
        )
        if estimator_instance.data_to_show:
            dic_to_put = {
                "t": t,
                "force_est": force_est.tolist(),
                "q_est": q_est.tolist(),
                "init_time_frame": absolute_time_received_s,
            }
            try:
                estimator_instance.plot_queue.get_nowait()
            except:
                pass
            estimator_instance.plot_queue.put_nowait(dic_to_put)

        time_to_get_data = time() - tic
        time_to_solve = sol.real_time_to_optimize
        time_tot = time_to_solve + time_to_get_data
        if estimator_instance.save_results:
            if not estimator_instance.save_all_frame:
                if mhe.kalman is not None:
                    mhe.kalman = np.append(
                        mhe.kalman,
                        x_ref_to_save[
                            :,
                            estimator_instance.frame_to_save: estimator_instance.frame_to_save
                            + estimator_instance.slide_size,
                        ],
                        axis=1,
                    )
                else:
                    mhe.kalman = x_ref_to_save[
                        :,
                        estimator_instance.frame_to_save: estimator_instance.frame_to_save
                        + estimator_instance.slide_size,
                    ]

                data_to_save = {
                    "time": time() - initial_time,
                    "X_est": np.concatenate((q_est, dq_est), axis=0),
                    "U_est": u_est,
                    "kalman": x_ref_to_save[
                        :,
                        estimator_instance.frame_to_save: estimator_instance.frame_to_save
                        + estimator_instance.slide_size,
                    ],
                    "f_est": force_est,
                    "none_conv_iter": stat,
                    "solver_options": estimator_instance.solver_options,
                }
                if t == 1:
                    data_to_save["Nmhe"] = estimator_instance.ns_mhe
                data_to_save["muscles_target"] = (
                    muscles_target_to_save[
                        :,
                        estimator_instance.frame_to_save: estimator_instance.frame_to_save
                        + estimator_instance.slide_size,
                    ]
                    if "muscle_target" in target.keys()
                    else 0
                )
                if estimator_instance.kin_data_to_track == "q":
                    kin_target = kin_target_to_save[
                        :,
                        estimator_instance.frame_to_save: estimator_instance.frame_to_save
                        + estimator_instance.slide_size,
                    ]

                else:
                    kin_target = kin_target_to_save[
                        :,
                        :,
                        estimator_instance.frame_to_save: estimator_instance.frame_to_save
                        + estimator_instance.slide_size,
                    ]

                if estimator_instance.use_torque:
                    data_to_save["tau_est"] = sol.controls["tau"][
                        :,
                        estimator_instance.frame_to_save: estimator_instance.frame_to_save
                        + estimator_instance.slide_size,
                    ]
            else:
                if mhe.kalman is not None:
                    mhe.kalman = np.append(mhe.kalman, x_ref_to_save, axis=1)
                else:
                    mhe.kalman = x_ref_to_save

                data_to_save = {
                    "time": time() - initial_time,
                    "X_est": np.concatenate((q_est, dq_est), axis=0),
                    "U_est": u_est,
                    "kalman": x_ref_to_save,
                    "f_est": force_est,
                    "none_conv_iter": stat,
                    "solver_options": estimator_instance.solver_options,
                }
                if t == 1:
                    data_to_save["Nmhe"] = estimator_instance.ns_mhe
                data_to_save["muscles_target"] = muscles_target_to_save if "muscle_target" in target.keys() else 0
                if estimator_instance.kin_data_to_track == "q":
                    kin_target = kin_target_to_save

                else:
                    kin_target = kin_target_to_save

                if estimator_instance.use_torque:
                    data_to_save["tau_est"] = sol.controls["tau"]

            data_to_save["kin_target"] = kin_target
            data_to_save["sol_freq"] = 1 / time_tot
            data_to_save["exp_freq"] = estimator_instance.exp_freq
            data_to_save["sleep_time"] = (1 / estimator_instance.exp_freq) - time_tot
            data_to_save["absolute_delay_tcp"] = absolute_delay_tcp
            data_to_save["absolute_time_receive"] = absolute_time_received_dic
            data_to_save["absolute_time_frame"] = absolute_time_frame
            if vicon_latency:
                data_to_save["vicon_latency"] = vicon_latency
            save_results(
                data_to_save,
                estimator_instance.current_time,
                estimator_instance.kin_data_to_track,
                estimator_instance.track_emg,
                estimator_instance.use_torque,
                estimator_instance.result_dir,
                file_name=estimator_instance.result_file_name,
            )
            if estimator_instance.print_lvl == 1:
                print(
                    f"Solve Frequency : {1 / time_tot} \n"
                    f"Expected Frequency : {estimator_instance.exp_freq}\n"
                    f"time to sleep: {(1 / estimator_instance.exp_freq) - time_tot}\n"
                    f"time to get data = {time_to_get_data}"
                )

        current_time = time() - tic
        time_tot = time_to_solve + current_time
        if 1 / time_tot > estimator_instance.exp_freq:
            sleep((1 / estimator_instance.exp_freq) - time_tot)
        estimator_instance.slide_size = tmp_slide_size

    if estimator_instance.test_offline:
        try:
            if target["kin_target"][1].shape[2] > estimator_instance.ns_mhe:
                return True
        except:
            if target["kin_target"][1].shape[1] > estimator_instance.ns_mhe:
                return True
        else:
            return False
    else:
        return True


class CustomMhe(MovingHorizonEstimator):
    """
    Class for the custom MHE.
    """

    def __init__(self, **kwargs):
        self.x_ref = None
        self.muscles_ref = None
        self.f_ext = None
        self.with_f_ext = False
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
        self.nlp[0].x_init.init[:, :] = np.concatenate(
            (
                sol.states["all"][:, self.slide_size:],
                np.repeat(sol.states["all"][:, -1][:, np.newaxis], self.slide_size, axis=1),
            ),
            axis=1,
        )
        return True

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        self.nlp[0].u_init.init[:, :] = np.concatenate(
            (
                sol.controls["all"][:, self.slide_size: -1],
                np.repeat(sol.controls["all"][:, -2][:, np.newaxis], self.slide_size, axis=1),
            ),
            axis=1,
        )
        if self.with_f_ext:
            self.nlp[0].u_init.init[-6:, :] = self.f_ext
        return True

    def advance_window_bounds_states(self, sol, **advance_options):
        self.nlp[0].x_bounds.min[:, 0] = sol.states["all"][:, self.slide_size]
        self.nlp[0].x_bounds.max[:, 0] = sol.states["all"][:, self.slide_size]
        return True

    def advance_window_bounds_controls(self, sol, **advance_options):
        if self.f_ext_as_constraints:
            self.nlp[0].u_bounds.min[-6:, 0] = self.f_ext[:, 0]
            self.nlp[0].u_bounds.max[-6:, 0] = self.f_ext[:, 0]
            self.nlp[0].u_bounds.min[-6:, 1] = self.f_ext[:, 1:-1]
            self.nlp[0].u_bounds.max[-6:, 1] = self.f_ext[:, 1:-1]
            self.nlp[0].u_bounds.min[-6:, -1] = self.f_ext[:, -1]
            self.nlp[0].u_bounds.max[-6:, -1] = self.f_ext[:, -1]
            return True
        else:
            return False

    def export_data(self, sol) -> tuple:
        return (
            sol.states["all"][:, self.frame_to_export: self.frame_to_export + 1],
            sol.controls["all"][:, self.frame_to_export: self.frame_to_export + 1],
        )

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
