import numpy as np
import datetime


def update_plot(estimator_instance, force_est: np.ndarray, q_est: np.ndarray, init_time: float = None):
    """
    Update the plot of the mhe.

    Parameters
    ----------
    estimator_instance: instance of the estimator class
        The estimator class.
    force_est: np.ndarray
        The estimated force.
    q_est: np.ndarray
        The estimated joint angles.
    init_time: float
        The initial time.
    """

    absolute_delay_plot = 0
    if estimator_instance.data_to_show.count("force") != 0:
        estimator_instance.force_to_plot = np.append(
            estimator_instance.force_to_plot[:, -estimator_instance.exp_freq - 1:], force_est, axis=1
        )
        estimator_instance.all_plot.update_plot_window(
            estimator_instance.all_plot.plot[0],
            data=estimator_instance.force_to_plot,
            app=estimator_instance.app_force,
            rplt=estimator_instance.rplt_force,
            box=estimator_instance.layout_force,
        )

        estimator_instance.count_p_f = 0
        estimator_instance.count_p_f += 1

    if estimator_instance.data_to_show.count("q") != 0:
        n_plot = 0 if not "force" in estimator_instance.data_to_show else 1
        estimator_instance.all_plot.update_plot_window(estimator_instance.all_plot.plot[n_plot], np.array(q_est)[:, -1])

    if init_time:
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
        absolute_delay_plot = absolute_time_received_s - init_time

    return np.round(absolute_delay_plot, 3)
