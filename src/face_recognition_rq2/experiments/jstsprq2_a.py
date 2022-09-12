import os.path
import sys
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jstsp_rq2_a_dict import d

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def far_n_perc(far, value):
    absolute_val_array = np.abs(far - value)
    smallest_difference_index = absolute_val_array.argmin()
    return smallest_difference_index


def most_sim_idx(far, frr, eer):
    curr_idx = 0
    diff = np.inf
    for el1, el2 in zip(far, frr):
        avg_err = np.average((np.abs(el1 - eer), np.abs(el2 - eer)))
        if avg_err <= diff:
            diff = avg_err
            curr_idx += 1
    return curr_idx


def get_eer_and_tars(cosines, group_ref, m):
    far_values = [0.01, 0.001, 0.0001, 0.00001]
    groups = list(set(group_ref))
    l = len(groups)

    fars = []
    frrs = []
    fixed_fars = []
    for grp_idx, group in enumerate(groups):
        subset = cosines[np.asarray(group_ref) == group]
        subset_norm = normalizeData(subset)
        fpr, tpr, thresholds = metrics.roc_curve(m[np.asarray(group_ref) == group], subset_norm, pos_label=1)
        idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
        if len(idxs) > 0:
            fpr = np.delete(fpr, idxs)
            tpr = np.delete(tpr, idxs)
            thresholds = np.delete(thresholds, idxs)
        fnr = 1 - tpr
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        except:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
        idx_far_frr = most_sim_idx(fpr, fnr, eer)
        indexes = [far_n_perc(fpr, v) for v in far_values]
        thresh = interp1d(fpr, thresholds)(eer)
        fars.append(fpr[idx_far_frr])
        frrs.append(fnr[idx_far_frr])
        fixed_fars.append([fnr[x] for x  in indexes])
    return fars, frrs, fixed_fars


def get_data(npy_file):
    results = npy_file[:, 0]
    m = npy_file[:, 1]
    group_ref = npy_file[:, 2].astype(np.int32).tolist()
    ref_id = npy_file[:, 4].astype(np.int32).tolist()
    return results, m, group_ref, ref_id


def get_graphs(data, dataset_name, mode):

    if mode == "genre":
        labels = ["Male", "Female"]
        colors = ["red", "blue"]
    elif mode == "ethnicity":
        labels = ["Asian", "Black", "Caucasian"]
        colors = ["red", "blue", "yellow"]
    elif mode == "both":
        labels = ["AM", "AW", "BM", "BW", "CM", "CW"]
        colors = ["red", "orange", "blue", "yellow", "violet", "black"]
    else:
        print("Wrong mode selected. Exiting...")
        sys.exit(1)


    label_model = ["HR_HR", "HR_LR", "LR_HR", "LR_LR"]
    fars_data, frrs_data, fixed_fars_data = [], [], []
    for d in data:
        results, m, group_ref, ref_id = get_data(d)
        fars, frrs, fixed_fars = get_eer_and_tars(results, group_ref, m)
        fars_data.append(fars)
        frrs_data.append(frrs)
        fixed_fars_data.append(fixed_fars)

    df_fars = pd.DataFrame(columns=["FAR", "Model-Dataset", "Group"])
    for i, e in enumerate(fars_data):
        for j, v in enumerate(e):
            df_fars = df_fars.append({"FAR": v, "Model-Dataset": label_model[i], "Group": labels[j]}, ignore_index=True)
    df_frrs = pd.DataFrame(columns=["FRR", "Model-Dataset", "Group"])
    for i, e in enumerate(frrs_data):
        for j, v in enumerate(e):
            df_frrs = df_frrs.append({"FRR": v, "Model-Dataset": label_model[i], "Group": labels[j]}, ignore_index=True)

    df_fixedfars = pd.DataFrame(columns=["FRR", "Model-Dataset", "Group", "@FAR"])
    far_values = [0.01, 0.001, 0.0001, 0.00001]
    for i, e in enumerate(fixed_fars_data):
        for j, v in enumerate(e):
            for k, fv in enumerate(far_values):
                df_fixedfars = df_fixedfars.append(
                    {"FRR": v[k], "Model-Dataset": label_model[i], "Group": labels[j], "@FAR": fv}, ignore_index=True)

    df_f_a = df_fixedfars.loc[df_fixedfars['@FAR'] == 0.01]
    df_f_b = df_fixedfars.loc[df_fixedfars['@FAR'] == 0.001]
    df_f_c = df_fixedfars.loc[df_fixedfars['@FAR'] == 0.0001]
    df_f_d = df_fixedfars.loc[df_fixedfars['@FAR'] == 0.00001]

    fig, ax = plt.subplots(3, 2, facecolor='white')
    fig.set_size_inches((20, 30))
    ax[0, 0].set_title("FAR @ EER")
    ax[0, 1].set_title("FRR @ EER")
    ax[1, 0].set_title("FRR @ FAR = 0.01")
    ax[1, 1].set_title("FRR @ FAR = 0.001")
    ax[2, 0].set_title("FRR @ FAR = 0.0001")
    ax[2, 1].set_title("FRR @ FAR = 0.00001")
    sns.barplot(data=df_fars, ax=ax[0, 0], hue='Group', y='FAR', x="Model-Dataset")
    sns.barplot(data=df_frrs, ax=ax[0, 1], hue='Group', y='FRR', x="Model-Dataset")
    sns.barplot(data=df_f_a, ax=ax[1, 0], hue='Group', y='FRR', x="Model-Dataset")
    sns.barplot(data=df_f_b, ax=ax[1, 1], hue='Group', y='FRR', x="Model-Dataset")
    sns.barplot(data=df_f_c, ax=ax[2, 0], hue='Group', y='FRR', x="Model-Dataset")
    sns.barplot(data=df_f_d, ax=ax[2, 1], hue='Group', y='FRR', x="Model-Dataset")
    #plt.show()
    bp = "resulsts_RQ2_a"
    if not os.path.exists(bp):
        os.makedirs(bp)
    sub = os.path.join(bp, dataset_name)
    if not os.path.exists(sub):
        os.makedirs(sub)
    plt.savefig(os.path.join(sub, "graphs.png"))


if __name__ == '__main__':
    import argparse
    """
    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--datasets_dict', metavar='path',
                            help='path to model')
        args = parser.parse_args()
    except Exception as e:
        print(e)
        sys.exit(1)
    ds = args.datasets_dict
    """
    ds = d
    for key, value in ds.items():
        HR_HR = np.load(value['HR_HR'])
        HR_LR = np.load(value['HR_LR'])
        LR_HR = np.load(value['LR_HR'])
        LR_LR = np.load(value['LR_LR'])
        data = [HR_HR, HR_LR, LR_HR, LR_LR]
        if key == "DiveFace":
            get_graphs(data=data, dataset_name=key, mode="both")
        if key in ["MAAD", "CelabA"]:
            get_graphs(data=data, dataset_name=key, mode="genre")
        if key in ["RFW", "BUPT"]:
            get_graphs(data=data, dataset_name=key, mode="ethnicity")
