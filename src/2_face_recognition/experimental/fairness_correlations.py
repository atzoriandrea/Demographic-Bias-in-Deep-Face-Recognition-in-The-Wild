import sys
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
from scipy.stats import kruskal
import matplotlib.ticker as tkr
pd.options.mode.chained_assignment = None
from sklearn import metrics
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from fairness_plots_across_resolutions import prepare_files, normalizeData, far_threshold_idx, most_sim_idx, get_data, get_g, get_e


def usability_at_security_level(cosines, group_ref, m, ref_id, values):
    far_values = values
    groups = list(set(group_ref))
    ref_id = np.asarray(ref_id)
    fixed_fars = []
    cosines = normalizeData(cosines)
    ts = [far_threshold_idx(normalizeData(cosines), m, v) for v in far_values]

    for grp_idx, group in enumerate(groups):
        ids = []
        subset = cosines[np.asarray(group_ref) == group]
        g_ids = ref_id[np.asarray(group_ref) == group]
        sub_m = m[np.asarray(group_ref) == group]
        ids_unique = np.unique(g_ids)
        for id in ids_unique:
            subsubset = subset[g_ids == id]
            fpr, tpr, thresholds = metrics.roc_curve(sub_m[g_ids == id], subsubset, pos_label=1)
            idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
            if len(idxs) > 0:
                fpr = np.delete(fpr, idxs)
                tpr = np.delete(tpr, idxs)
                thresholds = np.delete(thresholds, idxs)
            fnr = 1 - tpr
            ids.append([fnr[most_sim_idx(thresholds, x)] for x in ts])
        fixed_fars.append(ids)
    return fixed_fars


def get_params(task):
    metric = ["FRR", "FAR"] if task != "Identification" else ["FNIR", "FPIR"]
    fun = usability_at_security_level
    values = [0.01, 0.001, 0.0001] if task != "Identification" else [0.3, 0.2, 0.1]
    params = {
        "metric" : metric,
        "function" : fun,
        "values" : values
    }
    return params


def get_table_RQ1_AISC_ul(np_file_p, params, task):
    dataset_name = np_file_p.split("/")[-3]
    model_name = np_file_p.split("/")[-4]
    folder = os.path.join("tables", model_name, task, dataset_name)

    title = np_file_p.split("/")[-2]+"\n\n"
    print(title)
    u_s = params["metric"] #usability_security
    mod_n = title.strip().split("_")[0].capitalize()
    results, m, group_ref, ref_id = get_data(np.load(np_file_p))
    frrs = params["function"](results, group_ref, m, ref_id, params["values"])
    df_fixedfars = pd.DataFrame(columns=[u_s[0], "Group", "@"+u_s[1]])
    far_values = params["values"]
    labels = ["AM", "AW", "BM", "BW", "CM", "CW"]
    for j, v_l in enumerate(frrs):
        for v in v_l:
            for k, fv in enumerate(far_values):
                df_fixedfars = df_fixedfars.append(
                    {u_s[0]: v[k],  "Group": labels[j], "@"+u_s[1]: fv}, ignore_index=True)
    df_fixedfars["genre"] = df_fixedfars["Group"].apply(lambda x : get_g(x))
    df_fixedfars["ethnicity"] = df_fixedfars["Group"].apply(lambda x : get_e(x))
    df_fixedfars["cmp type"] = title
    df_fixedfars["Dataset"] = dataset_name
    return df_fixedfars


def get_table_single_attr_ul(list_of_files, params, task):
    dataset_name = list_of_files[0].split("/")[-3]
    model_name = list_of_files[0].split("/")[-4]
    folder = os.path.join("tables", model_name, task)
    u_s = params["metric"]
    df_fixedfars = pd.DataFrame(columns=[u_s[0], "Group", "@"+u_s[1], "cmp type"])
    far_values = params["values"]
     #usability_security
    for np_file_p in list_of_files:
        title = np_file_p.split("/")[-1].upper()+"\n\n"
        print(title)
        mod_n = title.strip().split("_")[0].capitalize()
        results, m, group_ref, ref_id = get_data(np.load(np_file_p))
        frrs = params["function"](results, group_ref, m, ref_id, params["values"])

        if len(frrs) == 3:
            labels = ["Asian", "Black", "Caucasian"]
        if len(frrs) == 2:
            labels = ["Man", "Woman"]
        for j, v_l in enumerate(frrs):
            for v in v_l:
                for k, fv in enumerate(far_values):
                    df_fixedfars = df_fixedfars.append(
                        {u_s[0]: v[k],  "Group": labels[j], "@"+u_s[1]: fv, "cmp type":title.split(".")[0]}, ignore_index=True)
    df_fixedfars = df_fixedfars.sort_values("Group", ascending=True)
    df_fixedfars["Dataset"] = dataset_name
    return df_fixedfars


def get_results(files_list, task):
    params = get_params(task)
    res = []
    DF = []
    for el in files_list:
        if type(el) == list:
            res.append(get_table_single_attr_ul(el, params, task))
            print(el[0].split("/")[-3])
        else:
            DF.append(get_table_RQ1_AISC_ul(el, params, task))
            print(el.split("/")[-3])
    res.append((pd.concat(DF)))
    return res


def plot_correlations(results, task):
    tot = pd.concat(results)
    tot["cmp type"] = tot["cmp type"].apply(lambda x: x.strip())
    tot = tot[tot['FNIR'].notna()]
    cbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    ds = ["DiveFace", "MAAD", "CelebA", "RFW", "BUPT"]
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(wspace=0, hspace=1)

    for d_id, d in enumerate(ds):  # ["Distributions"]:#, "Differences" ]:
        fig, ax = plt.subplots(1, 4, figsize=(25, 5), facecolor='white', sharey=True)
        sns.set(font_scale=1.0)
        sub = tot[tot["Dataset"] == d]
        sub["Dataset"] = sub["Dataset"].replace({"MAAD": "VGGFace2"})
        d = np.unique(sub["Dataset"])[0]
        cmps = sorted(list(set(sub["cmp type"])))
        groups = sorted(list(set(sub["Group"])))
        print(groups)
        for idx, c in enumerate(cmps):
            subsub = sub[sub["cmp type"] == c]
            frr_diff = np.zeros((len(groups), len(groups)))
            # far_diff = np.zeros((6,6))
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    frr_diff[i, j] = kruskal(subsub[subsub["Group"] == g1]["FNIR"].to_numpy()
                                             , subsub[subsub["Group"] == g2]["FNIR"].to_numpy())[1]
            ax[idx].set_title(d + " - " + c)
            sns.heatmap(data=np.round(frr_diff, 3),
                        annot=True,
                        square=True,
                        xticklabels=groups,
                        yticklabels=groups,
                        fmt="",
                        ax=ax[idx],
                        cbar=False,  # cbar_kws={"ticks": cbar_ticks, "format": formatter},
                        cmap=sns.color_palette("Blues", as_cmap=True), )
        plt.savefig(d + " Kruscal " + task + ".png", bbox_inches="tight")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--results_root', metavar='path',
                            help='path to model')
        args = parser.parse_args()
    except Exception as e:
        print(e)
        sys.exit(1)

    files = prepare_files(args.results_root)
    verification = sorted([x for x in files if "Verification" in x])
    identification = sorted([x for x in files if "Identification" in x])
    tasks = ["Verification", "Identification"]
    for files_sub, task in zip([verification, identification], tasks):
        params = get_params(task)
        results = get_results(files_sub, task)
        plot_correlations(results, task)



