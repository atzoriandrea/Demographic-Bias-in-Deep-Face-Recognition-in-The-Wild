import sys

import numpy as np
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
pd.options.mode.chained_assignment = None
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn import metrics
from itertools import islice, cycle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def most_sim_idx(ths, value):
    absolute_val_array = np.abs(ths - value)
    smallest_difference_index = absolute_val_array.argmin()
    return smallest_difference_index

def far_threshold_idx(cosines_norm, m, far_t, label =""):
    fpr, tpr, thresholds = metrics.roc_curve(m, cosines_norm, pos_label=1)
    idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
    if len(idxs) > 0:
        fpr = np.delete(fpr, idxs)
        tpr = np.delete(tpr, idxs)
        thresholds = np.delete(thresholds, idxs)
    fnr = 1 - tpr

    thresh = interp1d(fpr, thresholds)(far_t)#thresholds[index]
    print(label + " FNIR@FPIR="+str(far_t)+" : " + str(fnr[most_sim_idx(thresholds, thresh)]))

    return thresh

def usability_at_security_level(cosines, group_ref, m, values):
    far_values = values
    groups = list(set(group_ref))
    l = len(groups)
    fixed_fars = []
    #fpr_tot, tpr_tot, thresholds_tot = metrics.roc_curve(m, cosines, pos_label=1).
    cosines = normalizeData(cosines)
    ts = [far_threshold_idx(normalizeData(cosines),m, v) for v in far_values]

    for grp_idx, group in enumerate(groups):
        subset = cosines[np.asarray(group_ref) == group]
        #subset_norm = normalizeData(subset)
        fpr, tpr, thresholds = metrics.roc_curve(m[np.asarray(group_ref) == group], subset, pos_label=1)
        idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
        if len(idxs) > 0:
            fpr = np.delete(fpr, idxs)
            tpr = np.delete(tpr, idxs)
            thresholds = np.delete(thresholds, idxs)
        fnr = 1 - tpr


        #print(fnr[x])
        fixed_fars.append([fnr[most_sim_idx(thresholds, x)] for x  in ts])
    return fixed_fars



def get_data(npy_file):
    results = npy_file[:, 0]
    m = npy_file[:, 1]
    group_ref = npy_file[:, 2].astype(np.int32).tolist()
    ref_id = npy_file[:, 4].astype(np.int32).tolist()
    return results, m, group_ref, ref_id
def get_g(v):
    if "M" in v:
        return  "Male"
    elif "W" in v:
        return "Female"
    return None
def get_e(v):
    if "A" in v:
        return "Asian"
    elif "B" in v:
        return "Black"
    elif "C" in v:
        return "Caucasian"
    return None

def get_files_full_path(rootdir):

    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".npy"):
                paths.append(os.path.join(root, file))
    return paths

def prepare_files(rootdir):
    files_full = sorted(get_files_full_path(rootdir))
    new_filelist = []
    i = 0
    # for _ in range(len(files_full)):
    while i + 4 <= len(files_full):
        if "DiveFace" not in files_full[i]:
            new_filelist.append(files_full[i:i + 4])
            i += 4
        else:
            new_filelist.append(files_full[i])
            i += 1
    return new_filelist


def get_params(task):
    metric = ["FRR", "FAR"] if task != "Identification" else ["FNIR", "FPIR"]
    fun = usability_at_security_level #get_frr_at_fars if task != "Identification" else get_fnir_at_fpirs
    values = [0.01, 0.001, 0.0001] if task != "Identification" else [0.3, 0.2, 0.1]
    params = {
        "metric" : metric,
        "function" : fun,
        "values" : values
    }
    return params


def get_table_RQ1_AISC(np_file_p, params, task):
    dataset_name = np_file_p.split("/")[-3]
    model_name = np_file_p.split("/")[-4]
    folder = os.path.join("tables", model_name, task, dataset_name)

    title = np_file_p.split("/")[-2]+"\n\n"
    print(title)
    u_s = params["metric"] #usability_security
    mod_n = title.strip().split("_")[0].capitalize()
    results, m, group_ref, ref_id = get_data(np.load(np_file_p))
    frrs = params["function"](results, group_ref, m, params["values"])
    df_fixedfars = pd.DataFrame(columns=[u_s[0], "Group", "@"+u_s[1]])
    far_values = params["values"]
    labels = ["AM", "AW", "BM", "BW", "CM", "CW"]
    for j, v in enumerate(frrs):
        for k, fv in enumerate(far_values):
            df_fixedfars = df_fixedfars.append(
                {u_s[0]: v[k],  "Group": labels[j], "@"+u_s[1]: fv}, ignore_index=True)
    df_fixedfars["genre"] = df_fixedfars["Group"].apply(lambda x : get_g(x))
    df_fixedfars["ethnicity"] = df_fixedfars["Group"].apply(lambda x : get_e(x))
    #df_00001 = df_fixedfars[df_fixedfars["@FAR"] == 0.00001]
    df_0001 = df_fixedfars[df_fixedfars["@"+u_s[1]] == far_values[2]]
    df_001 = df_fixedfars[df_fixedfars["@"+u_s[1]] == far_values[1]]
    df_01 = df_fixedfars[df_fixedfars["@"+u_s[1]] == far_values[0]]
    df_01 = df_01.groupby(["@"+u_s[1], "ethnicity",'genre']).mean().pivot_table(index='ethnicity', columns=["@"+u_s[1], 'genre'])
    df_001 = df_001.groupby(["@"+u_s[1], "ethnicity",'genre']).mean().pivot_table(index='ethnicity', columns=["@"+u_s[1], 'genre'])
    df_0001 = df_0001.groupby(["@"+u_s[1], "ethnicity",'genre']).mean().pivot_table(index='ethnicity', columns=["@"+u_s[1], 'genre'])
    #df_00001 = df_00001.groupby(["@FAR", "ethnicity",'genre']).mean().pivot_table(index='ethnicity', columns=["@FAR", 'genre'])
    df_01.loc[:,(u_s[0],str(far_values[0]),'M+F')] = df_01.mean(numeric_only=True, axis=1)
    df_001.loc[:,(u_s[0],str(far_values[1]),'M+F')] = df_001.mean(numeric_only=True, axis=1)
    df_0001.loc[:,(u_s[0],str(far_values[2]),'M+F')] = df_0001.mean(numeric_only=True, axis=1)
    #df_00001.loc[:,('FRR',"0.00001",'M+F')] = df_00001.mean(numeric_only=True, axis=1)
    total = df_01.join(df_001.join(df_0001, on="ethnicity"), on='ethnicity')
    total.loc['A+B+C']= total.mean(numeric_only=True, axis=0)
    print(total.round(decimals=3).to_latex())
    df_fixedfars["Model"] = mod_n
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, np_file_p.split("/")[-2]+".tex"), "w") as texfile:
        texfile.write(total.round(decimals=3).to_latex())
    df_fixedfars["cmp type"] = title
    df_fixedfars["Dataset"] = dataset_name
    return df_fixedfars


def get_table_single_attr(list_of_files, params, task):
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
        frrs = params["function"](results, group_ref, m, params["values"])

        if len(frrs) == 3:
            labels = ["Asian", "Black", "Caucasian"]
        if len(frrs) == 2:
            labels = ["Man", "Woman"]
        for j, v in enumerate(frrs):
            for k, fv in enumerate(far_values):
                df_fixedfars = df_fixedfars.append(
                    {u_s[0]: v[k],  "Group": labels[j], "@"+u_s[1]: fv, "cmp type":title.split(".")[0]}, ignore_index=True)
    df_fixedfars = df_fixedfars.sort_values("Group", ascending=True)
    df_0001 = df_fixedfars[df_fixedfars["@"+u_s[1]] == far_values[2]]
    df_001 = df_fixedfars[df_fixedfars["@"+u_s[1]] == far_values[1]]
    df_01 = df_fixedfars[df_fixedfars["@"+u_s[1]] == far_values[0]]

    if len(frrs) == 3:
        df_01 = df_01.groupby(["@"+u_s[1], "cmp type",'Group']).mean().pivot_table(index='Group', columns=["@"+u_s[1], 'cmp type'])
        df_001 = df_001.groupby(["@"+u_s[1], "cmp type",'Group']).mean().pivot_table(index='Group', columns=["@"+u_s[1], 'cmp type'])
        df_0001 = df_0001.groupby(["@"+u_s[1], "cmp type",'Group']).mean().pivot_table(index='Group', columns=["@"+u_s[1], 'cmp type'])
    if len(frrs) == 2:
        df_01 = df_01.groupby(["@"+u_s[1], "cmp type",'Group']).mean().pivot_table(index='Group', columns=["@"+u_s[1], 'cmp type']).iloc[::-1]
        df_001 = df_001.groupby(["@"+u_s[1], "cmp type",'Group']).mean().pivot_table(index='Group', columns=["@"+u_s[1], 'cmp type']).iloc[::-1]
        df_0001 = df_0001.groupby(["@"+u_s[1], "cmp type",'Group']).mean().pivot_table(index='Group', columns=["@"+u_s[1], 'cmp type']).iloc[::-1]
    total = df_01.join(df_001.join(df_0001, on="Group"), on='Group')
    total.loc['A+B+C']= total.mean(numeric_only=True, axis=0)
    print(total.round(decimals=3).to_latex())
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, dataset_name+".tex"), "w") as texfile:
        texfile.write(total.round(decimals=3).to_latex())
    df_fixedfars["Dataset"] = dataset_name
    return df_fixedfars


def get_results(files_list, task):
    params = get_params(task)
    res = []
    DF = []
    for el in files_list:
        if type(el) == list:
            res.append(get_table_single_attr(el, params, task))
            print(el[0].split("/")[-3])
        else:
            DF.append(get_table_RQ1_AISC(el, params, task))
            print(el.split("/")[-3])
    res.append((pd.concat(DF)))
    return res


def get_plots(results, params, task):
    std_devs = []  # pd.DataFrame(columns=["std. dev", "Dataset", "@"+params["metric"][1], "Train-test Quality"])
    ds = ["DiveFace", "VGGFace2", "CelebA", "RFW", "BUPT"]


    ##################### STD VARS PLOT #############################
    for i, r in enumerate(results):
        r["cmp type"] = r["cmp type"].apply(lambda x: x.strip())
        r["Dataset"] = r["Dataset"].replace({"CelabA": "CelebA"})
        devs = r[[params["metric"][0], "@" + params["metric"][1], "cmp type"]].groupby(
            ["cmp type", "@" + params["metric"][1]]).std().reset_index()
        devs["Dataset"] = np.unique(r["Dataset"])[0]
        std_devs.append(devs.sort_values("@" + params["metric"][1],
                                         ascending=False))
    devs_tot = pd.concat(std_devs).rename(columns={params["metric"][0]: 'std. dev', 'cmp type': 'Train-test Quality'})
    devs_tot["@" + params["metric"][1]] = devs_tot["@" + params["metric"][1]].apply(lambda x: str(x * 100) + "%")
    fig, ax = plt.subplots(5, 1, figsize=(10, 20), facecolor='white')
    fig.subplots_adjust(hspace=.4)
    sns.set(style="whitegrid")
    for i, d in enumerate(ds):
        sub = devs_tot[devs_tot["Dataset"] == d]
        ax[i].title.set_text(d)
        sns.barplot(data=sub[["Train-test Quality", "std. dev", "@" + params["metric"][1]]], x="Train-test Quality",
                    y="std. dev", hue="@" + params["metric"][1], ax=ax[i])
    plt.savefig(task + " std vars.png", bbox_inches="tight")

    ##################### MEANS PLOT ##################################

    means = []
    for i, r in enumerate(results):
        r["cmp type"] = r["cmp type"].apply(lambda x: x.strip())
        r["Dataset"] = r["Dataset"].replace({"CelabA": "CelebA"})
        m = r[[params["metric"][0], "@" + params["metric"][1], "cmp type"]].groupby(
            ["cmp type", "@" + params["metric"][1]]).mean().reset_index()
        m["Dataset"] = np.unique(r["Dataset"])[0]
        means.append(m.sort_values("@" + params["metric"][1],
                                   ascending=False))  # = std_devs.append({"std. dev": devs[params["metric"][0]].to_frame(), "@"+params["metric"][1]:  devs["@"+params["metric"][1]].to_frame(), "Train-test Quality": devs["cmp type"].apply(lambda x: x.strip()).to_frame(), "Dataset": devs["Dataset"].to_frame()}, ignore_index=True)
    means_tot = pd.concat(means).rename(columns={params["metric"][0]: 'mean', 'cmp type': 'Train-test Quality'})
    means_tot["@" + params["metric"][1]] = means_tot["@" + params["metric"][1]].apply(lambda x: str(x * 100) + "%")

    fig, ax = plt.subplots(5, 1, figsize=(10, 20), facecolor='white')
    fig.subplots_adjust(hspace=.4)
    sns.set(style="whitegrid")
    for i, d in enumerate(ds):
        sub = means_tot[means_tot["Dataset"] == d]
        ax[i].title.set_text(d)
        sns.barplot(data=sub[["Train-test Quality", "mean", "@" + params["metric"][1]]], x="Train-test Quality",
                    y="mean", hue="@" + params["metric"][1], ax=ax[i])
    plt.savefig(task + " means.png", bbox_inches="tight")

    ################## PERF ACROSS GROUPS #################
    group_means = []
    for i, r in enumerate(results):
        r["cmp type"] = r["cmp type"].apply(lambda x: x.strip())
        r["Dataset"] = r["Dataset"].replace({"CelabA": "CelebA"})
        m2 = r[[params["metric"][0], "@" + params["metric"][1], "cmp type", "Group"]].groupby(
            ["cmp type", "@" + params["metric"][1], "Group"]).mean().reset_index()
        m2["Dataset"] = np.unique(r["Dataset"])[0]
        group_means.append(m2.sort_values("@" + params["metric"][1],
                                          ascending=False))  # = std_devs.append({"std. dev": devs[params["metric"][0]].to_frame(), "@"+params["metric"][1]:  devs["@"+params["metric"][1]].to_frame(), "Train-test Quality": devs["cmp type"].apply(lambda x: x.strip()).to_frame(), "Dataset": devs["Dataset"].to_frame()}, ignore_index=True)
    means_tot2 = pd.concat(group_means).rename(columns={'cmp type': 'Train-test Quality'})


    fig, ax = plt.subplots(5, 1, figsize=(10, 20), facecolor='white')
    fig.subplots_adjust(hspace=.4)
    sns.set(style="whitegrid")
    red_p = (0.07, 0.098, 0.29)
    yellow_p = (0.29, 0.32, 0.67)
    green_p = (0.725, 0.764, 0.93)
    sns.set(style="whitegrid", font_scale=.8)
    p = params["values"]
    for i, d in enumerate(ds):
        sub = means_tot2[means_tot2["Dataset"] == d]
        labels = np.unique(sorted(sub["Group"]))
        if labels.shape[0] < 6:
            sub["Group"] = sub["Group"].apply(lambda x: x[0][0])
            labels = np.unique(sorted(sub["Group"]))
        df_0001 = sub[sub["@" + params["metric"][1]] == p[2]]
        df_001 = sub[sub["@" + params["metric"][1]] == p[1]]
        df_01 = sub[sub["@" + params["metric"][1]] == p[0]]
        top_bar = mpatches.Patch(color=red_p, label=params["metric"][1] + " " + str(p[2] * 100) + "%")
        mid_bar = mpatches.Patch(color=yellow_p, label=params["metric"][1] + " " + str(p[1] * 100) + "%")
        bottom_bar = mpatches.Patch(color=green_p, label=params["metric"][1] + " " + str(p[0] * 100) + "%")
        ########## PLOT 2 ######################
        bar3 = sns.barplot(data=df_0001.sort_values(['Train-test Quality', "Group"]), y=params["metric"][0],
                           x='Train-test Quality', hue='Group', palette=[red_p], ax=ax[i])
        bar2 = sns.barplot(data=df_001.sort_values(['Train-test Quality', "Group"]), y=params["metric"][0],
                           x='Train-test Quality', hue='Group', palette=[yellow_p], ax=ax[i])
        bar1 = sns.barplot(data=df_01.sort_values(['Train-test Quality', "Group"]), y=params["metric"][0],
                           x='Train-test Quality', hue='Group', palette=[green_p], ax=ax[i])
        ax[i].title.set_text(d)
        ax[i].legend(handles=[top_bar, mid_bar, bottom_bar])
        # ["AM", "AW", "BM", "BW", "CM", "CW"]
        for i, l in enumerate(labels):
            bar3.bar_label(bar3.containers[i], labels=[l for _ in range(4)], padding=3)

        # pad the spacing between the number and the edge of the figure
        bar3.margins(y=0.1)
    plt.savefig(task + " perf_across_groups.png", bbox_inches="tight")


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
        get_plots(results, params, task)



