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
from casadi import MX, vertcat
import numpy as np


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
    ConfigureProblem.configure_new_variable("f_ext", ["mx", "my", "mz", "fx", "fy", "fz"],
                                            ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_torque_driven)


def get_objectives(weigths, nb_q, nb_markers, n_shooting, with_f_ext=False, track_previous=False, target=None):
    f_ext = np.zeros((6, n_shooting + 1))
    # target = np.zeros((3, nb_markers, n_shooting + 1))
    q_init = np.zeros((nb_q, n_shooting + 1))
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=weigths["qdot"],
                            multi_thread=False,
                            index=list(range(abs(10 - nb_q), nb_q)), derivative=False)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=weigths["tau"],
                            multi_thread=False,
                            index=list(range(abs(10 - nb_q), nb_q)))
    if with_f_ext:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="f_ext", weight=weigths["f_ext"],
                                target=f_ext[:, :n_shooting], node=Node.ALL_SHOOTING, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=weigths["markers"],
                            target=target[..., :n_shooting + 1],
                            node=Node.ALL, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=weigths["q"], multi_thread=False,
                            index=list(range(abs(10 - nb_q), nb_q)))

    if track_previous:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=weigths["q_prev"],
                                target=q_init[: nb_q, :n_shooting + 1],
                                node=Node.ALL, multi_thread=False)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", weight=weigths["qdot_prev"],
                                target=q_init[nb_q: nb_q * 2, :n_shooting + 1],
                                node=Node.ALL, multi_thread=False)

    if nb_q > 10:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, multi_thread=False,
                                index=list(range(4, 5)), quadratic=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1000, multi_thread=False,
                            index=list(range(9, 10)), quadratic=True, derivative=False)

    return objective_functions


def prepare_ocp(biorbd_model_path, final_time, n_shooting,
                use_sx=False,
                n_threads=1,
                mhe=False,
                kin_init=None,
                target=None,
                f_ext=None,
                with_f_ext=False,
                weights=None,
                track_previous=False):
    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max, tau_init = -10000, 10000, 0
    objective_functions = get_objectives(weights, bio_model.nb_q, bio_model.nb_markers, n_shooting, with_f_ext,
                                         track_previous=track_previous, target=target
                                         )
    dynamics = DynamicsList()
    if with_f_ext:
        dynamics.add(custom_configure,
                     dynamic_function=custom_torque_driven,
                     expand_dynamics=True)
    else:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", min_bound=[-1000] * bio_model.nb_tau, max_bound=[1000] * bio_model.nb_tau)

    x_init = InitialGuessList()
    x_init.add("q", kin_init[: bio_model.nb_q, :n_shooting + 1], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", kin_init[bio_model.nb_q:, :n_shooting + 1], interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    u_bounds = BoundsList()

    u_init = InitialGuessList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model.nb_tau, max_bound=[tau_max] * bio_model.nb_tau)
    u_init.add("tau", [tau_init] * bio_model.nb_tau)
    if with_f_ext:
        u_bounds.add("f_ext", min_bound=[-12000] * 6, max_bound=[12000] * 6)
        u_init.add("f_ext", f_ext[:, :n_shooting], interpolation=InterpolationType.EACH_FRAME)
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
        ), bio_model
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
        ), bio_model


def get_solver_options(solver):
    sol_dict = {"solver_first_iter": None, "solver": solver}
    solver_options = {
        "qp_solver_iter_max": 1000,
    }
    if isinstance(solver, Solver.ACADOS):
        for key in solver_options.keys():
            sol_dict["solver"].set_option_unsafe(val=solver_options[key], name=key)
        sol_dict["solver"].set_convergence_tolerance(1e-4)
        from copy import copy
        sol_dict["solver"] = copy(sol_dict["solver"])
        sol_dict["solver"].set_print_level(1)
        sol_dict["solver"].set_qp_solver("PARTIAL_CONDENSING_HPIPM")  # PARTIAL_CONDENSING_OSQP PARTIAL_CONDENSING_HPIPM
        sol_dict["solver"].set_integrator_type("IRK")
        sol_dict["solver"].set_maximum_iterations(100)
        sol_dict["solver"].set_convergence_tolerance(1e-5)

    elif isinstance(solver, Solver.IPOPT):
        sol_dict["solver"].set_hessian_approximation("exact")
        # sol_dict["solver"].set_limited_memory_max_history(50)
        sol_dict["solver"].set_maximum_iterations(50)
        sol_dict["solver"].set_print_level(5)
        sol_dict["solver"].set_tol(1e-5)
        sol_dict["solver"].set_linear_solver("ma57")
        from copy import copy
        sol_dict["solver_first_iter"] = copy(sol_dict["solver"])
        sol_dict["solver_first_iter"].set_maximum_iterations(100)
        sol_dict["solver_first_iter"].set_tol(1e-5)

    else:
        raise NotImplementedError("Solver not recognized")
    return sol_dict

def get_update_function(markers_init, f_ext, with_f_ext, track_previous, kin_init, n_shooting, model, ocp):
    def update_functions(mhe, t, _):
        def target_mark(i: int):
            return markers_init[:, :, i: i + n_shooting + 1]

        def target_f_ext(i: int):
            return f_ext[:, i: i + n_shooting + 1]

        if with_f_ext:
            mhe.update_objectives_target(target=target_f_ext(t), list_index=2)
            mhe.update_objectives_target(target=target_mark(t), list_index=3)
            if track_previous:
                if ocp.sol is not None:
                    previous_sol = ocp.sol.decision_states(to_merge=SolutionMerge.NODES)
                q_to_track = previous_sol[
                    "q"] if ocp.sol is not None else kin_init[:model.nbQ(), t:t + n_shooting + 1]
                qdot_to_track = previous_sol[
                    "qdot"] if ocp.sol is not None else kin_init[model.nbQ():, t:t + n_shooting + 1]
                mhe.update_objectives_target(target=q_to_track, list_index=5)
                mhe.update_objectives_target(target=qdot_to_track, list_index=6)
        else:
            mhe.update_objectives_target(target=target_mark(t), list_index=2)
            if track_previous:
                previous_sol = ocp.sol.decision_states(to_merge=SolutionMerge.NODES)
                q_to_track = previous_sol[
                    "q"] if ocp.sol is not None else kin_init[:model.nbQ(), t:t + n_shooting + 1]
                qdot_to_track = previous_sol[
                    "qdot"] if ocp.sol is not None else kin_init[model.nbQ():, t:t + n_shooting + 1]
                mhe.update_objectives_target(target=q_to_track, list_index=5)
                mhe.update_objectives_target(target=qdot_to_track, list_index=6)
        return t < kin_init.shape[1] - (n_shooting + 1)
    return update_functions