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

def get_frr_at_fars(cosines, group_ref, m):
    far_values = [0.01, 0.001, 0.0001]
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

def get_fnir_at_fpirs(cosines, group_ref, m):
    far_values = [0.3, 0.2, 0.1]
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


def get_table_RQ1_AISC(np_file_p, params):
    dataset_name = np_file_p.split("/")[-3]
    model_name = np_file_p.split("/")[-4]
    folder = os.path.join("tables", model_name, task, dataset_name)

    title = np_file_p.split("/")[-2]+"\n\n"
    print(title)
    u_s = params["metric"] #usability_security
    mod_n = title.strip().split("_")[0].capitalize()
    results, m, group_ref, ref_id = get_data(np.load(np_file_p))
    frrs = params["function"](results, group_ref, m)
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