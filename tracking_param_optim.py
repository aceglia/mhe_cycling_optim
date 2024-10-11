from biosiglive import load
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':

    muscle_names = ['TrapeziusScapula_M',  # 0
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

    participants = [f"P{i}" for i in range(10, 17)]
    #participants = ["P10"]
    participants.pop(participants.index("P12"))
    participants.pop(participants.index("P15"))
    participants.pop(participants.index("P16"))
    all_params = np.zeros((len(participants), 2, 35))
    all_params_id = np.zeros((len(participants), 2, 35))

    result_dir = "/mnt/shared/Projet_hand_bike_markerless/optim_params/results"
    for p,  part in enumerate(participants):
        param_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param.bio"
        data_tmp = load(param_path)
        all_params[p, 0, :], all_params[p, 1, :] = np.array(data_tmp["p"][0])[:, 0],  np.array(data_tmp["p"][1])[:, 0]
        param_path_id = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_id.bio"
        data_tmp_id = load(param_path_id)
        all_params_id[p, 0, :], all_params_id[p, 1, :] = np.array(data_tmp_id["p"][0])[:, 0],  np.array(data_tmp_id["p"][1])[:, 0]
    all_params = np.round(all_params, 3)
    all_params_id = np.round(all_params_id, 3)

    import matplotlib.pyplot as plt
    import numpy as np
    fig_name = ["F_iso", "l_optim"]
    for p in range(2):
        # Example data (35 muscles and 8 participants)
        participant_data = all_params[:, p, :].T # Replace this with your actual data
        participant_data_id = all_params_id[:, p, :].T # Replace this with your actual data

        # Plotting
        plt.figure(fig_name[p])

        # Number of participants
        n_participants = participant_data.shape[1]

        # Width of each bar
        bar_width = 0.1

        # Indices for muscle positions
        indices = np.arange(len(muscle_names))
        # Width of each bar and the space between muscles
        bar_width = 0.1
        muscle_spacing = 0.3

        # Indices for muscle positions with added spacing
        indices = np.arange(len(muscle_names)) * (bar_width * 2 * n_participants + muscle_spacing)
        indices = np.linspace(0, len(muscle_names), len(muscle_names))

        # Generate a color map for the participants
        colors = plt.cm.get_cmap('tab10', n_participants)

        # Plot bars for participants
        count = 0
        for i in range(n_participants):
            # Separate data into above and below 1
            above_one = np.clip(participant_data[:, i] - 1, 0, None)  # Values above 1
            below_one = np.clip(participant_data[:, i] - 1, None, 0)  # Values below 1
            above_one_id = np.clip(participant_data_id[:, i] - 1, 0, None)  # Values above 1
            below_one_id = np.clip(participant_data_id[:, i] - 1, None, 0)  # Values below 1

            # Plot above 1 values
            plt.bar(indices + count, above_one, width=bar_width, label=f'Participant {i + 1}', color=colors(i))
            plt.bar(indices + count + bar_width, above_one_id, width=bar_width, label=f'Participant {i + 1}_id', color=colors(i), hatch="*")

            # Plot below 1 values (in the negative direction)
            plt.bar(indices + count, below_one, width=bar_width, color=colors(i))
            plt.bar(indices + count + bar_width, below_one_id, width=bar_width, color=colors(i), hatch="*")
            count += (bar_width * 2)

        # Set x-ticks and labels
        idxs = indices + (len(participants) * bar_width * 2)/2
        for m in range(len(muscle_names)):
            plt.axvline(idxs[m], linestyle='--', alpha=0.2, c="k")
        plt.xticks(indices + (len(participants) * bar_width * 2)/2, muscle_names, rotation=90)
        y_ticks = np.arange(0.4-1, 2.8-1, 0.1)  # Define range for y-ticks
        y_labels = [f'{1 + tick:.1f}' for tick in y_ticks]  # Create labels centered on 1
        plt.yticks(y_ticks, y_labels)
        # Center y-axis at 1
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=1.8, color='black', linestyle='--')
        plt.axhline(y=-0.6, color='black', linestyle='--')

        # Labels and title
        plt.ylabel('Deviation from 1')
        plt.title('Bidirectional Plot of Optimized Parameters for Muscles Across Participants')

        # Add legend
        plt.legend()

        # Display the plot
        plt.tight_layout()
    plt.show()

