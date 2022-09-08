import pandas as pd
import json
import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def counter(json_file, keys, npy_file, csv_file):
    def compute_EER(cosines_norm, matches):
        fpr, tpr, thresholds = metrics.roc_curve(matches, cosines_norm, pos_label=1)
        idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
        if len(idxs) > 0:
            fpr = np.delete(fpr, idxs)
            tpr = np.delete(tpr, idxs)
            thresholds = np.delete(thresholds, idxs)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        # print(eer)
        return thresh

    errors_by_group = [[] for _ in range(6)]

    results = np.load(npy_file)
    results[:, 0] = normalizeData(results[:, 0])
    t = compute_EER(results[:, 0], results[:, 1])
    # print(t)
    with open(csv_file, "r") as csv:
        lines = csv.readlines()
    couples = []
    for i, line in enumerate(lines):
        data = line.split(",")
        try:
            r = [0, 0, 0, 0, i, 0, 0]
            c1 = "/".join(data[0].split("/")[-2:])
            attributes = [json_file[c1][k] for k in keys]
            image = c1
            group = int(results[i, 2])
            row = [image]
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
            r[6] = results[i, 1]
            row.extend(r)
            errors_by_group[group].append(row)
        except:
            pass
    return errors_by_group


def scale(values):
    A = values.iloc[0] / values.iloc[2]
    B = values.iloc[1] / values.iloc[2]
    value = values.iloc[2]
    return A, B, value


def merge_data(json_file_name, results_file_name, couples_csv_file_name, embeddings_file_name, dataset_files_txt, regroup=True,
               num=5):
    with open(json_file_name, 'r') as fp:
        f = json.load(fp)
    keys = list(f[list(f.keys())[0]].keys())
    errors = counter(json_file=f, keys=keys, npy_file=results_file_name, csv_file=couples_csv_file_name)
    soft_attributes_list = []
    other_info = ['FR', 'FA', 'TR', 'TA', 'row_in_file', 'similarity', 'Genuine']
    cols = ['Identity'] + keys + other_info
    for grp_idx, group in enumerate(errors):
        df0 = pd.DataFrame(group, columns=cols)
        df0 = df0.drop(columns=['row_in_file'])
        soft_attributes_list.append(df0)
    soft_attributes = pd.concat(soft_attributes_list)

    embeddings = np.load(embeddings_file_name)

    # reading dataset file (containing images full paths)
    with open(dataset_files_txt, "r") as dftn:
        names = dftn.readlines()
        for i, n in enumerate(names):
            names[i] = "/".join(n.strip().split("/")[-2:])
    d = {'Names': names, 'Embeddings': embeddings.tolist()}
    ids_emb = pd.DataFrame(d)
    ids_emb['Names'] = ids_emb['Names'].apply(lambda x: x.split(" ")[0]).astype(str)
    ids_emb['Embeddings'] = ids_emb['Embeddings'].apply(lambda x: np.asarray(x))

    df_tot = pd.merge(ids_emb, soft_attributes, left_on='Names', right_on='Identity', how='right').drop(columns='Names')
    if regroup:
        df_tot['Identity'] = df_tot['Identity'].apply(lambda x: x.split("/")[0])
    else:
        df_tot['user'] = df_tot['Identity']
        df_tot['Identity'] = df_tot['Identity'].apply(lambda x: x.split("/")[0])
    genuines = df_tot[df_tot['Genuine'] == 1].groupby('Identity').head(num).reset_index(drop=True)
    imposters = df_tot[df_tot['Genuine'] == 0].groupby('Identity').head(num).reset_index(drop=True)
    balanced_tot = pd.concat([genuines, imposters])

    genuines_similarities = balanced_tot[balanced_tot['Genuine'] == 1][['Identity', 'similarity']].groupby(
        'Identity').mean().rename(columns={'similarity': 'similarity_genuine'})
    imposter_similarities = balanced_tot[balanced_tot['Genuine'] == 0][['Identity', 'similarity']].groupby(
        'Identity').mean().rename(columns={'similarity': 'similarity_imposter'})
    genuines_counter = df_tot[df_tot['Genuine'] == 1].groupby('Identity').count().iloc[:, 0].to_frame(name='sim_num')
    imposter_counter = df_tot[df_tot['Genuine'] == 0].groupby('Identity').count().iloc[:, 0].to_frame(name='imp_num')
    rates_tot = df_tot[['Identity', 'FR', 'FA', 'TR', 'TA']].groupby('Identity').sum()
    counters = pd.merge(genuines_counter, imposter_counter, on='Identity')
    rates_tot_w_ratios = pd.merge(rates_tot, counters, on='Identity')
    rates_tot_w_ratios['FA'] = rates_tot_w_ratios['FA'] / rates_tot_w_ratios['imp_num']
    rates_tot_w_ratios['TR'] = rates_tot_w_ratios['TR'] / rates_tot_w_ratios['imp_num']
    rates_tot_w_ratios['FR'] = rates_tot_w_ratios['FR'] / rates_tot_w_ratios['sim_num']
    rates_tot_w_ratios['TA'] = rates_tot_w_ratios['TA'] / rates_tot_w_ratios['sim_num']
    rates_tot_w_ratios.drop(columns=['imp_num', 'sim_num'], inplace=True)
    #rates_tot_w_ratios[['FA', 'TR', 'sim_num']].apply(lambda x : x[[]])
    used_keys = ['Identity', 'Embeddings', 'head_roll', 'head_pitch', 'blur', 'moustache', 'smile', 'age', 'beard',
                 'sideburns', 'noise', 'head_yaw',
                 'exposure', 'mouth_occluded', 'mask', 'headWear', 'forehead_occluded', 'lip_makeup', 'eye_occluded',
                 'glasses', 'eye_makeup']
    if regroup:
        grouped_balanced_tot = balanced_tot[used_keys].groupby('Identity').agg(list)
    else:
        grouped_balanced_tot = df_tot[used_keys]
    non_stacked = pd.merge(pd.merge(genuines_similarities, imposter_similarities, on='Identity'), rates_tot_w_ratios, on='Identity')
    full_ds = pd.merge(grouped_balanced_tot, non_stacked, on='Identity')

    return full_ds
