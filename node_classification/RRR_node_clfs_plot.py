import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import numpy as np


def main():
    axis_label_font_size = 12
    dataset_num = "4"
    graph_name = "prostate_subgraph_p3"  # Current oprions: PokeReport, MergedPokemon, HierarchicalPokemon
    dataset_path = f"./data/RRR_node_clf/rrr_curve_{graph_name}_{dataset_num}"
    colors = ['g', 'y', 'b', 'c', 'r', 'orange', 'k', 'm']

    hide_x = False


    in_labels = ["Flexible walk", "Flexible walk with RTM",
                 "Combined walk", "Combined walk with RTM",
                 "Fixed walk", "Fixed walk with RTM"]
    in_tables = [f"{dataset_path}/FlexWalcDepth0-10_DT_means.xlsx",
                 f"{dataset_path}/FlexWalcDepth0-10_DT_RTM_means.xlsx",
                 f"{dataset_path}/CombWalcDepth0-10_DT_means.xlsx",
                 f"{dataset_path}/CombWalcDepth0-10_DT_RTM_means.xlsx",
                 f"{dataset_path}/FixWalcDepth0-10_DT_means.xlsx",
                 f"{dataset_path}/FixWalcDepth0-10_DT_RTM_means.xlsx"
                 ]


    # collection all .xlsx files in the directory:
    #in_tables = [f"{dataset_path}/{f}" for f in os.listdir(dataset_path) if f.endswith(".xlsx")]
    #in_labels = [f.split("/")[-1].replace(".xlsx", "") for f in in_tables]

    #in_labels = ["Frequency based DT", "MINDWALC DT"]
    #in_labels = ["Flexible walking depth", "Combined walking depth", "Fixed walking depth"]

    ###### plot f1-test curve ######
    for i, in_table in enumerate(in_tables):
        # load xlsx with pandas:
        df = pd.read_excel(in_table)

        f_one_v = df["f1_mean"]
        x = df["x"] * 100
        f_one_std_v = df["f1_std"]

        # plot x against f1 as curve:
        plt.plot(x, f_one_v, label=in_labels[i], color=colors[i], marker='o')
        # plt.plot(x, f_one_v, label=in_labels[i], color=colors[i])
        plt.legend(loc="lower left")

        if not hide_x:
            plt.xlabel("Instance Knowledge Degradation (%)", fontsize=axis_label_font_size)
        plt.ylabel("F1-Score", fontsize=axis_label_font_size)
        plt.title("F1-Score" if hide_x else "F1-Score vs Instance Knowledge Degradation")

        plt.fill_between(x, [f_one_v[i] + f_one_std_v[i] for i in range(len(x))],
                         [f_one_v[i] - f_one_std_v[i] for i in range(len(x))], facecolor=colors[i], alpha=0.1)

        if hide_x:
            plt.xticks([])

    # plt.show()
    plt.grid()
    plt.savefig(f"{dataset_path}/f1_vs_feature_destruction.png", dpi=300)

    ###### plot f1-train curve ######
    plt.clf()
    for i, in_table in enumerate(in_tables):
        # load xlsx with pandas:
        df = pd.read_excel(in_table)

        f_one_v = df["f1_mean_train"]
        x = df["x"] * 100
        f_one_std_v = df["f1_std_train"]

        # plot x against f1 as curve:
        plt.plot(x, f_one_v, label=in_labels[i], color=colors[i], marker='o')
        # plt.plot(x, f_one_v, label=in_labels[i], color=colors[i])
        plt.legend(loc="lower left")

        if not hide_x:
            plt.xlabel("Instance Knowledge Degradation (%)", fontsize=axis_label_font_size)
        plt.ylabel("Train-F1-Score", fontsize=axis_label_font_size)
        plt.title(plt.title("Train-F1-Score" if hide_x else "Train-F1-Score vs Instance Knowledge Degradation"))

        plt.fill_between(x, [f_one_v[i] + f_one_std_v[i] for i in range(len(x))],
                         [f_one_v[i] - f_one_std_v[i] for i in range(len(x))], facecolor=colors[i], alpha=0.1)

        if hide_x:
            plt.xticks([])

    # plt.show()
    plt.grid()
    plt.savefig(f"{dataset_path}/f1_vs_feature_destruction_train.png", dpi=300)

    ######## plot complexity ##########
    plt.clf()
    for i, in_table in enumerate(in_tables):
        # load xlsx with pandas:
        df = pd.read_excel(in_table)

        # plot x against f1 as curve:
        plt.plot(df["x"] * 100, df["max_tree_depth_mean"], label=in_labels[i], color=colors[i], marker='o')
        # plt.plot(x, f_one_v, label=in_labels[i], color=colors[i])
        plt.legend(loc="upper left")  #lower right

        if not hide_x:
            plt.xlabel("Instance Knowledge Degradation (%)", fontsize=axis_label_font_size)
        plt.ylabel("Decision Tree depth", fontsize=axis_label_font_size)
        plt.title("Decision Tree complexity" if hide_x else "Decision Tree complexity vs Instance Knowledge Degradation")

        plt.fill_between(df["x"] * 100,
                         [df["max_tree_depth_mean"][i] + df["max_tree_depth_std"][i] for i in range(len(df["x"]))],
                         [df["max_tree_depth_mean"][i] - df["max_tree_depth_std"][i] for i in range(len(df["x"]))],
                         facecolor=colors[i], alpha=0.1)

        if hide_x:
            plt.xticks([])

    # plt.show()
    plt.grid()
    plt.savefig(f"{dataset_path}/tree_depth_vs_feature_destruction.png", dpi=300)

    plt.clf()
    for i, in_table in enumerate(in_tables):
        # load xlsx with pandas:
        df = pd.read_excel(in_table)

        # plot x against f1 as curve:
        plt.plot(df["x"] * 100, df["node_count_mean"], label=in_labels[i], color=colors[i], marker='o')
        # plt.plot(x, f_one_v, label=in_labels[i], color=colors[i])
        plt.legend(loc="lower right")

        if not hide_x:
            plt.xlabel("Instance Knowledge Degradation (%)", fontsize=axis_label_font_size)
        plt.ylabel("Decision Tree node count", fontsize=axis_label_font_size)
        plt.title("Decision Tree complexity" if hide_x else "Decision Tree complexity vs Instance Knowledge Degradation")

        plt.fill_between(df["x"] * 100,
                         [df["node_count_mean"][i] + df["node_count_std"][i] for i in range(len(df["x"]))],
                         [df["node_count_mean"][i] - df["node_count_std"][i] for i in range(len(df["x"]))],
                         facecolor=colors[i], alpha=0.1)

        if hide_x:
            plt.xticks([])
    plt.grid()
    plt.savefig(f"{dataset_path}/node_count_vs_feature_destruction.png", dpi=300)

    ######## plot walking depths ##########

    plt.clf()
    # get all subfolders in the directory dataset_path:
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    subfolders.sort(key=lambda x: float(x.split("RRR_")[2]))

    # get all curve_names:
    #clf_names = [str(os.path.basename(f.path)) for f in os.scandir(subfolders[0]) if f.is_dir()]
    clf_names = [str(os.path.basename(f)).replace("_means.xlsx", "") for f in in_tables]

    for i_clf, clf_name in enumerate(clf_names):
        # collect all walking depths:
        walk_depths = {}
        for subfolder in subfolders:
            rrr_value = float(subfolder.split("RRR_")[2])
            walk_depths[rrr_value] = []
            path_to_decision_trees = f"{subfolder}/{clf_name}/trees"
            # get all files ending with "_named.gv" in the directory path_to_decision_trees:
            files = [f for f in os.listdir(path_to_decision_trees) if f.endswith("_named.gv")]
            for tree_file in files:
                with open(f"{path_to_decision_trees}/{tree_file}", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "d = " in line:
                            try: # int convertion fails at flexible walks, where the walk depth is a min-max tuple...
                                d_value = int(line.split("d = ")[1].split('",')[0])
                            except:
                                continue
                            walk_depths[rrr_value].append(d_value)

        skip = False
        for k in walk_depths.keys():
            if len(walk_depths[k]) == 0:  # skip curve plot if no walking depths are found
                skip = True
                break
            avrg = sum(walk_depths[k]) / len(walk_depths[k])
            std = sum([(d - avrg) ** 2 for d in walk_depths[k]]) / len(walk_depths[k])
            min_max = (min(walk_depths[k]), max(walk_depths[k]))
            walk_depths[k] = (avrg, std, min_max)

        if skip:
            continue

        x = list(walk_depths.keys())
        x_plot = [float(k) * 100 for k in walk_depths.keys()]
        y = [walk_depths[k][0] for k in x]
        #y = np.array(y)
        yerr = [walk_depths[k][1] for k in x]
        #yerr = np.array(yerr)
        min_max = [walk_depths[k][2] for k in x]

        # plot x against svg depth as curve:
        plt.plot(x_plot, y, label=in_labels[i_clf], color=colors[i_clf], marker='o')
        # plt.plot(x, f_one_v, label=in_labels[i], color=colors[i])
        plt.legend(loc="upper right")

        if not hide_x:
            plt.xlabel("Instance Knowledge Degradation (%)", fontsize=axis_label_font_size)
        plt.ylabel("Average walking depths", fontsize=axis_label_font_size)
        plt.title("Walking depth" if hide_x else "Walking depths vs Instance Knowledge Degradation")

        # fill with +- std:
        plt.fill_between(x_plot, [y[i] + yerr[i] for i in range(len(x))], [y[i] - yerr[i] for i in range(len(x))],
                         facecolor=colors[i_clf], alpha=0.1)

        # fill with min max:
        #plt.fill_between(x_plot, [mm[0] for mm in min_max], [mm[1] for mm in min_max], facecolor=colors[i_clf], alpha=0.1)

        if hide_x:
            plt.xticks([])

        # plt.show()
    plt.grid()
    plt.savefig(f"{dataset_path}/walkd_depth_vs_feature_destruction.png", dpi=300)

    return 0


if __name__ == '__main__':
    main()
