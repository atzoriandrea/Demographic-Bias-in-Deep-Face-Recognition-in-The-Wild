import json

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import stats
from sklearn import metrics

continuouous_features = [0, 1, 2, 3, 4, 8, 9, 10, 16, 17, 18]
binary_features = [5, 6, 7, 11, 12, 13, 14, 15]


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def counter(json_file, npy_file, csv_file, keys):
    def compute_EER(cosines_norm, matches):
        fpr, tpr, thresholds = metrics.roc_curve(matches, cosines_norm, pos_label=1)
        idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
        if len(idxs) > 0:
            fpr = np.delete(fpr, idxs)
            tpr = np.delete(tpr, idxs)
            thresholds = np.delete(thresholds, idxs)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        print(eer)
        return thresh

    errors_by_group = [[] for _ in range(6)]

    results = np.load(npy_file)
    results[:, 0] = normalizeData(results[:, 0])
    t = compute_EER(results[:, 0], results[:, 1])
    print(t)
    with open(csv_file, "r") as csv:
        lines = csv.readlines()
    couples = []
    for i, line in enumerate(lines):
        data = line.split(",")
        try:
            r = [0, 0, 0, 0, i, 0]
            c1 = "/".join(data[0].split("/")[-2:])
            attributes = [json_file[c1][k] for k in keys]
            identity = c1.split("/")[0]
            group = int(results[i, 2])
            row = [identity]
            row.extend(attributes)
            ####################
            # counter data format : FR, FA, TR, TA
            ####################
            if results[i, 0] >= t and results[i, 1] == 1:
                r[3] = 1
            if results[i, 0] < t and results[i, 1] == 1:
                r[0] = 1
            if results[i, 0] >= t and results[i, 1] == 0:
                r[1] = 1
            if results[i, 0] < t and results[i, 1] == 0:
                r[2] = 1
            r[5] = results[i, 0]
            row.extend(r)
            errors_by_group[group].append(row)
        except:
            pass
    return errors_by_group


def get_data_means(errors, keys):
    data_means = []
    other_info = ['FR', 'FA', 'TR', 'TA', 'row_in_file', 'similarity']
    genre_eth = {0: ["Male", "Asian"], 1: ["Female", "Asian"], 2: ["Male", "Black"], 3: ["Female", "Black"],
                 4: ["Male", "Caucasian"], 5: ["Female", "Caucasian"]}
    cols = ['Identity'] + keys + other_info
    for grp_idx, group in enumerate(errors):
        df0 = pd.DataFrame(group, columns=cols)
        df0 = df0.drop(columns=['row_in_file'])
        df_feat = df0[list(set(cols).difference(other_info).difference([keys[x] for x in binary_features]))]
        df_feat_mean = df_feat.groupby(['Identity']).mean()
        try:
            df_feat2 = df0[
                list(set(cols).difference(other_info).difference([keys[x] for x in continuouous_features]))].drop(
                columns=['group'])
        except:
            df_feat2 = df0[list(set(cols).difference(other_info).difference([keys[x] for x in continuouous_features]))]
        # df_feat2 = df_feat2.replace({True: 1.0, False: 0.0})
        df_feat2_mean = df_feat2.groupby(['Identity']).agg(lambda x: stats.mode(x)[0])
        df_data = df0[['Identity', 'FR', 'FA', 'TR', 'TA']]
        df_similarities = df0[['Identity', 'similarity']]
        df_similarities_list_genuine = df_similarities.groupby('Identity')['similarity'].apply(list).apply(
            lambda x: np.mean(x[:5])).to_frame()
        df_similarities_list_imposter = df_similarities.groupby('Identity')['similarity'].apply(list).apply(
            lambda x: np.mean(x[5:])).to_frame()
        df_similarities_list = df_similarities_list_genuine.join(df_similarities_list_imposter, on='Identity',
                                                                 lsuffix='_genuine', rsuffix='_imposter')
        df_data_sum = df_data.groupby(['Identity']).sum()
        df_tot = df_feat_mean.join(df_data_sum, on='Identity')
        df_tot = df_tot.join(df_feat2_mean, on='Identity')
        df_tot = df_tot.join(df_similarities_list, on='Identity')
        df_tot['group'] = grp_idx
        df_tot[['genre', 'ethnicity']] = genre_eth[grp_idx]
        df_tot['FA'] = df_tot['FA'] / 50
        df_tot['TR'] = df_tot['TR'] / 50
        df_tot['FR'] = df_tot['FR'] / 5
        df_tot['TA'] = df_tot['TA'] / 5
        data_means.append(df_tot)
    return pd.concat(data_means)
