from biosiglive import load
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    data = load("/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/results/results_w9/data_pedalage_1_result_duration_0.08.bio")
    X = np.linspace(0, 8, 9)
    num_frame = 1
    nom = ['clav_RotX', 'clav_RotY', 'scap_RotX', 'scap_RotY', 'scap_RotZ', 'hum_RotX', 'hum_RotY', 'hum_RotZ', 'elbow_flexion_RotZ']
    print(data)
    fig, ax = plt.subplots()
    ax.scatter(X, data['X_est'][:9, num_frame])
    for i in range(len(nom)):
        ax.annotate(nom[i], (X[i], data['X_est'][i, num_frame]))
    ax.scatter(X, data['kalman'][:9, num_frame], c='r')
    for i in range(len(nom)):
        ax.annotate(nom[i], (X[i], data['kalman'][i, num_frame]))
    plt.show()