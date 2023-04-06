import numpy as np
import biorbd
from biosiglive import save, load
import csv
import os

if __name__ == '__main__':
    model_path = "/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/results/wu_gauche_cycling_pos_scaled_3.bioMod"
    model = biorbd.Model(model_path)
    file_path = "/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/mhe_cycling_optim/data_gen/saves/pedalage_3_proc.bio"
    data =load(file_path)
    q = load(file_path)["kalman"]
    external_forces = [[]]
    n_shooting = q.shape[1]
    datas = []
    n_debut = 5000
    n_fin = n_debut + n_shooting
    with open(
            '/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/sandbox/pedalage/pedalage 3/Results-pedalage_3_001.lvm',
            'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            row_bis = [float(i) for i in row]
            datas.append(row_bis)
    datas = np.array(datas)
    force_x = datas[n_debut:n_fin, 21]
    force_y = datas[n_debut:n_fin, 22]
    force_z = datas[n_debut:n_fin, 23]
    moment_x = datas[n_debut:n_fin, 24]
    moment_y = datas[n_debut:n_fin, 25]
    moment_z = datas[n_debut:n_fin, 26]
    for i in range(n_shooting):

        A = [0, -0.068095000000000003, 0, 1]
        B = [0, 0, 0, 1]

        RT = model.allGlobalJCS(q[:9, i])[-1].to_array().T
        RT2 = model.allGlobalJCS(q[:9, i])[25].to_array().T

        A = np.dot(A, RT)
        B = np.dot(B, RT2)
        vecteur_AB = A[:3]-B[:3]
        # force = [6000, 6000, 6000]
        force = np.array([force_x[i], force_y[i], force_z[i]]).T
        moment_initial = np.array([moment_x[i], moment_y[i], moment_z[i]]).T
        moment_initial_2 = moment_initial + np.cross(vecteur_AB, force)
        moment_final = moment_initial_2 + np.cross(B[:3], force)
        external_forces_calc = np.array([moment_final[0], moment_final[1], moment_final[2], force[0], force[1], force[2]])[:, np.newaxis]
        external_forces[0].append(external_forces_calc)

    external_forces = np.array(external_forces)

    data["f_ext"] = external_forces[:, :, :, 0]
    if os.path.isfile(file_path):
                os.remove(file_path)
    save(data, file_path)