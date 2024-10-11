import numpy as np
import pandas

from biosiglive import load
import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def create_pd(mat_p):
    all_pd_p = []
    for part in range(mat_p.shape[0]):
        all_pd_tmp = []
        for m in range(mat_p.shape[2]):
            pd_p_iso = pd.DataFrame()
            #pd_p_iso["muscles"] = sum([[muscle_names[i]] * n_batch * mat_p.shape[0] for i in range(mat_p.shape[2])], [])
            pd_p_iso["cycles"] = sum([[f"cycle_{i}"] * n_batch  for i in range(mat_p.shape[1])], [])
            final_list = []
            for k in range(mat_p.shape[1]):
                final_list.append(sum([mat_p[part, k, m, :].tolist()], []))
            pd_p_iso["p"] = sum(final_list, [])
            all_pd_tmp.append(pd_p_iso)
        all_pd_p.append(all_pd_tmp)

    return all_pd_p

def plot_param(data, muscle_to_plot, participants, muscle_names, plot_name="plot", color_palette=0, save_fig=True):
    # fig = plt.figure(num=plot_name, constrained_layout=False)
    # subplots = fig.subplots(len(muscle_to_plot), len(participants), sharex=True,
    #                         sharey=True)
    def _plot_ridgeline(data, plot_name, color_palette):
        sn.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        if color_palette == 1:
            pal = sn.cubehelix_palette(10, rot=.25, light=.7)
        else:
            pal = sn.cubehelix_palette(10, rot=-.25, light=.7)


        fg = sn.FacetGrid(data, row="cycles", hue="cycles", palette=pal, aspect=8, height=1, col="muscles")
        fg.map(sn.kdeplot, "p",
              bw_adjust=2,
                clip_on=False,
              fill=True, alpha=1, linewidth=1.5
               #, common_norm=True
               )
        fg.map(sn.kdeplot, "p", clip_on=False, color="w", lw=1.5
               #, common_norm=True
        , bw_adjust=2
        )
        #fg.map(plt.axvline, x=1, ls='--', c='k')
        fg.refline(y=0, linewidth=0.8, linestyle="-", color=None, clip_on=False)
        for a in range(fg.axes.shape[1]):
            #fg.axes[-1][a].axvline(x=1, c="k", ymin=0, ymax=0.5, ls="--", alpha=0.5)
            fg.axes[-1][a].axvline(x=1, c="k", ymin=0, ymax=1, ls="--", alpha=0.5)
            #fg.axes[0][a].set_ylim((0, 15))
        #fg.axes[0][a].axvline(x=1, c="k", ymin=0, ymax=1, ls="--", alpha=0.5)

        #fg.refline(x=1, linewidth=0.3, linestyle="-", color="k", clip_on=False)



        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)


        #fg.map(label, "p")
        fg.figure.subplots_adjust(hspace=-0.9, wspace=0.2)
        #fg.figure.subplots_adjust(hspace=-0.9)

        fg.set_titles("")
        fg.set(yticks=[], ylabel="")
        if color_palette == 0:
            fg.set(xlim=(0.5, 2.5))
        else:
            fg.set(xlim=(0.5, 2))
        fg.set(ylim=(0, 7))

        fg.despine(bottom=True, left=True)
        if save_fig:
            plt.savefig("figs/" + plot_name + ".svg", format="svg")
        #plt.show()
    count = 0


    #for i in range(len(muscle_names)):
    #    if muscle_names[i] in muscle_to_plot:
    for p in range(len(participants)):
        pd_tmp = pandas.DataFrame()
        pd_tmp["cycles"] = sum([[f"cycle_{i}"] * n_batch for i in range(len(nodes))] * len(muscle_to_plot), [])
        pd_tmp["names"] = sum([[f'P'] * n_batch] * len(nodes) * len(muscle_to_plot), [])
        #pd_tmp["part"] = sum([[f'part_{i}'] * n_batch * len(nodes) for i in range(len(participants))], [])
        pd_tmp["muscles"] = sum([[muscle_names[i]] * n_batch * len(nodes) for i in range(len(muscle_names)) if muscle_names[i] in muscle_to_plot], [])

        #pd_tmp["p"] = sum([list(data[p][i]["p"]) for p in range(len(participants))], [])
        list_all_p = []
        count = 0
        for m in range(len(muscle_names)):
            if muscle_names[m] in muscle_to_plot:
                list_all_p += sum([data[p, k, m, :].tolist() for k in range(len(nodes))], [])
        pd_tmp["p"] = list_all_p

        # ax = subplots[count, p]
        #
        # if count == 0:
        #     ax.set_title(f"Participant {p+1}")
        # if p ==0:
        #     ax.set_ylabel(muscle_to_plot[count] + "\n\n" + "Factor", rotation=90)
        # if count == len(muscle_to_plot) - 1:
        #     ax.set_xticklabels([1, 2, 3, 4, 5, 6])
        #     ax.set_xlabel("Cycles")
        #
        # else:
        #     ax.set_xticklabels([])
        #     #ax.set_xticks([])
        # ax.grid()

        #ax.axhline(y=1, c="k", alpha=0.3)
        #ax.axvlines(x=[0,1,2,3,4,5], c="k")


        # sn.kdeplot(
        #     data=pd_p[i], x="p", hue="cycles",
        #     multiple="stack",
        #     fill=True,
        #     palette="crest",
        #     alpha=.5, linewidth=0.2,
        #     legend=legend
        # )
        _plot_ridgeline(pd_tmp, plot_name + "_" + participants[p], color_palette)
        # pd_tmp = pandas.DataFrame()
        # pd_tmp["cycles"] = sum([[f"cycle_{i}"] * n_batch for i in range(len(nodes))] * 2, [])
        # pd_tmp["names"] = sum([[f'P'] * n_batch] * len(nodes) + [[f'l'] * n_batch] * len(nodes), [])
        # pd_tmp["p"] = sum([list(data[p][i]["p"]) + list(data_bis[p][i]["p"])], [])
        #sn.boxplot(data=pd_tmp, x="cycles", y="p", hue="names", ax=ax, legend=False, palette=["b", "r"])

        #plt.ylim((0.5, 2.5))
        #plt.xlim((0.5, 2.5))
        #plt.margins(x=0)
        count +=1

if __name__ == '__main__':
    muscle_names= ['TrapeziusScapula_M',  # 0
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
    muscle_to_plot = [
        # 'TrapeziusScapula_M',  # 0
                   #'TrapeziusScapula_S',  # 1
                   # 'TrapeziusScapula_I',  # 2
                   'Rhomboideus_S',  # 3
                   #'Rhomboideus_I',  # 3
                   'LevatorScapulae',  # 4
                   #'PectoralisMinor',  # 5
                   # 'TrapeziusClavicle_S',  # 6
                   # 'SerratusAnterior_I',  # 7
                   #'SerratusAnterior_M',  # 7
                   # 'SerratusAnterior_S',  # 7
                   #'Coracobrachialis',  # 8
                   'DeltoideusScapula_P',  # 9
                   # 'DeltoideusScapula_M',  # 10
                   #'TeresMajor',  # 11
                   # 'Infraspinatus_I',  # 12
                   #'Infraspinatus_S',  # 12
                   # 'TeresMinor',  # 13
                   # 'Subscapularis_S',  # 14
                   'Subscapularis_M',  # 14
                   #'Subscapularis_I',  # 14
                   # 'Supraspinatus_P',  # 15
                   #'Supraspinatus_A',  # 15
                   # 'DeltoideusClavicle_A',  # 16
                   # 'PectoralisMajorClavicle_S',  # 17
                   # 'LatissimusDorsi_S',  # 18
                   'LatissimusDorsi_M',  # 18
                   # 'LatissimusDorsi_I',  # 18
                   #'PectoralisMajorThorax_I',  # 19
                   # 'PectoralisMajorThorax_M',  # 19
                   # "BRD",
                   # "PT",
                   # "PQ"
                   #'TRI_long',  # 20
                    #'TRI_lat',  # 20
                   #'TRI_med',  # 20
                   # 'BIC_long',  # 21
                   #"'BIC_brevis',
                      ]  # 21
    #muscle_to_plot = muscle_names

    participants = [f"P{i}" for i in range(10, 15)]
    if "P12" in participants:
        participants.pop(participants.index("P12"))
    #participants = ["P10", "P11"]
    trials = ["gear_20"]
    nodes = [1, 2, 3, 4, 5, 6,]

    n_batch = 10
    pd_p_iso = pd.DataFrame()
    pd_p_iso["muscles"] = sum([[muscle_names[i]] * n_batch * len(participants) for i in range(len(muscle_names))], []) * len(nodes)
    pd_p_iso["cycles"] = sum([[f"cycle_{i}"] * n_batch * len(muscle_names) * len(participants) for i in range(len(nodes))], [])
    pd_l_iso = pd.DataFrame()
    pd_l_iso["muscles"] = sum([[muscle_names[i]] * n_batch * len(participants) for i in range(len(muscle_names))], []) * len(nodes)
    pd_l_iso["cycles"] = sum([[f"cycle_{i}"] * n_batch * len(muscle_names) * len(participants) for i in range(len(nodes))], [])
    all_p = np.zeros((len(participants), len(nodes),  len(muscle_names), n_batch))
    all_l = np.zeros((len(participants), len(nodes),  len(muscle_names), n_batch))
    # p_iso_all_node = np.zeros((len(muscle_names) * n_batch * len(nodes) * len(participants)))
    from biosiglive import save
    for p, part in enumerate(participants):
        for t, node in enumerate(nodes):
            param_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_gear_20_fd_{node}_test_quad.bio"
            data_tmp = load(param_path, merge=False)

            # os.remove(f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_gear_20_fd_{node}_test_quad.bio")
            # for i in range(7):
            #     save(add_data=True, data_dict=data_tmp[i], data_path=f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/result_optim_param_gear_20_fd_{node}_test_quad.bio")

            n_batch_tmp = n_batch
            p_iso = []
            _ = [p_iso.append(np.array(data_tmp[i]["p"][0]).tolist()) for i in range(n_batch_tmp)]
            final_list_p = []
            for n in range(len(muscle_names)):
                final_list_p.append([p_iso[i][n] for i in range(n_batch_tmp)])
            l_optim = []
            _ = [l_optim.append(np.array(data_tmp[i]["p"][1]).tolist()) for i in range(n_batch_tmp)]
            final_list_l = []
            for n in range(len(muscle_names)):
                final_list_l.append([l_optim[i][n] for i in range(n_batch_tmp)])
            final_list_p = np.array(final_list_p)
            final_list_p[final_list_p == 0] = 'nan'
            final_list_l = np.array(final_list_l)
            final_list_l[final_list_l == 0] = 'nan'
            centered_list_p = [final_list_p[i] - np.mean(final_list_p[i]) for i in range(len(final_list_p))]
            centered_list_l = [final_list_l[i] - np.mean(final_list_l[i]) for i in range(len(final_list_l))]
            centered_list_l = final_list_l
            centered_list_p = final_list_p
            all_p[p, t, :, :] = np.array(centered_list_p).reshape(len(muscle_names), n_batch)
            all_l[p, t, :, :] = np.array(centered_list_l).reshape(len(muscle_names), n_batch)
    all_muscles_p = [] #np.zeros((len(muscle_names) * n_batch * len(nodes) * len(participants)))
    all_muscles_l = []# np.zeros((len(muscle_names) * n_batch * len(nodes) * len(participants)))
    pd_p = create_pd(all_p)
    pd_l = create_pd(all_l)
    sn.set_theme(style="ticks", palette="pastel")
    plot_param(all_p, muscle_to_plot, participants, muscle_names, plot_name="p_optim", color_palette=0, save_fig=True)
    plot_param(all_l, muscle_to_plot, participants, muscle_names, plot_name="l_optim", color_palette=1, save_fig=True)
    #plt.show()
