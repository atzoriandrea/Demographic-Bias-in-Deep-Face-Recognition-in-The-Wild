import argparse
import os
import sys

from get_data_means import *
import copy
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import json
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def get_pattern(p_value):
    if p_value < 0.01:
        new_value = "\\"
    elif p_value < 0.05:
        new_value = "o"
    else:
        new_value = ""
    return new_value


def get_mean_stddev(m):
    mean_stddev = np.zeros((2, m.shape[-1]))
    mean_stddev[0, :] = np.mean(m, axis=0)
    mean_stddev[1, :] = np.std(m, axis=0)
    return mean_stddev


def z_score_normalize(data_ori, mean_stddev):
    data = copy.deepcopy(data_ori)
    for i in range(data.shape[-1]):
        if mean_stddev[1, i] != 0:
            data[:, i] = (data[:, i] - mean_stddev[0, i]) / mean_stddev[1, i]
        else:
            print("Attribute %d has mean %f and std dev %f" % (i, mean_stddev[0, i], mean_stddev[1, i]))
    return data


def get_r2_and_coeffs(r2, coeff, models_names, features_list):
    df_list1, df_list2 = [], []
    for i, scores in enumerate(r2):
        data = pd.DataFrame({'R2 Scores': scores})
        data['Model'] = models_names[i]
        df_list1.append(data)
    R2_df = pd.concat(df_list1)

    for i, coe in enumerate(coeff):
        d = {}
        for idx, f in enumerate(features_list):
            d[f] = coe[:, idx]
        data = pd.DataFrame(d)
        data['Model'] = models_names[i]
        df_list2.append(data)
    COEFF_df = pd.concat(df_list2)
    return R2_df, COEFF_df


def plot_correlations(model_data, models_names, selected_measure, dataset_name):
    features_list = ['genre', 'ethnicity', 'age', 'moustache', 'beard', 'sideburns', 'eye_makeup', 'lip_makeup',
                     'headWear', 'glasses', 'head_roll', 'head_yaw', 'head_pitch', 'forehead_occluded', 'eye_occluded',
                     'mouth_occluded', 'exposure', 'blur', 'noise', 'smile']
    df = pd.DataFrame(columns=['Correlation', 'Model', 'Feature', 'PValue'])
    for i, md in enumerate(model_data):
        md['genre'] = md['genre'].replace({"Male": 0, "Female": 1})
        md['ethnicity'] = md['ethnicity'].replace({"Asian": 0, "Black": 1, "Caucasian": 2})
        for feature in features_list:
            r = stats.pearsonr(md[feature], md[selected_measure])
            pv = get_pattern(r[1])
            df = df.append({'Correlation': r[0], 'Model': models_names[i], 'Feature': feature, 'PValue': r[1],
                            'PValueTxt': pv}, ignore_index=True)
    ax2 = sns.barplot(data=df, y='PValue', x='Feature', hue='Model')
    fig = plt.figure(figsize=(30, 8), facecolor='white')
    plt.rcParams.update({'font.size': 22})
    sns.set_style("whitegrid")
    # plt.ylim((-0.16, 0.05))
    ax1 = sns.barplot(data=df, y='Correlation', x='Feature', hue='Model')
    plt.legend(ncol=5)
    p1 = ax1.patches
    p2 = ax2.patches
    for bar1, bar2 in zip(p1, p2):
        bar1.set_hatch(get_pattern(bar2.get_height()))

    plt.xticks(rotation=25)
    folder_base_name = "Correlations"
    folder_name = folder_base_name if dataset_name == "" else "_".join([folder_base_name, dataset_name])
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    plt.savefig(os.path.join(folder_name, "_".join(["Correlations", selected_measure + "R.png"])), bbox_inches='tight')


def plot_r2(R2_df, dataset_name):
    folder_base_name = "R2_Scores"
    folder_name = folder_base_name if dataset_name == "" else "_".join([folder_base_name, dataset_name])
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    sns.set_style("whitegrid")
    sns.boxplot(data=R2_df, y='R2 Scores', x='Model')
    plt.xticks(rotation=25)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    plt.savefig(os.path.join(folder_name, "_".join([folder_name, selected_measure + "R.png"])), bbox_inches='tight')


def plot_coeffs(COEFF_df, dataset_name):
    folder_base_name = "Feature_Weights"
    folder_name = folder_base_name if dataset_name == "" else "_".join([folder_base_name, dataset_name])
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    plt.rcParams.update({'font.size': 12})
    sns.set_palette("magma")
    sns.set_style("whitegrid")
    data = COEFF_df.groupby('Model').mean().T
    ax = sns.heatmap(data=data)
    plt.xticks(rotation=25)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    plt.savefig(os.path.join(folder_name, "_".join([folder_name, selected_measure + "R.png"])), bbox_inches='tight')


def plot_feature_weights(model_data, models_names, selected_measure, iterations, n_samples, dataset_name):
    R2 = []
    COEFF = []
    features_list = ['genre', 'ethnicity', 'age', 'moustache', 'beard', 'sideburns', 'eye_makeup', 'lip_makeup',
                     'headWear', 'glasses', 'head_roll', 'head_yaw', 'head_pitch', 'forehead_occluded', 'eye_occluded',
                     'mouth_occluded', 'exposure', 'blur', 'noise', 'smile']
    for i, md in enumerate(model_data):
        md['genre'] = md['genre'].replace({"Male": 0, "Female": 1})
        md['ethnicity'] = md['ethnicity'].replace({"Asian": 0, "Black": 1, "Caucasian": 2})
        R2_scores = np.zeros((iterations,))
        coefficients = np.zeros((iterations, len(features_list)))
        print(models_names[i])
        for e in range(iterations):
            training = md.sample(n=n_samples)
            X = training[features_list].to_numpy().astype(np.float32)
            Y = training[selected_measure].to_numpy().astype(np.float32)
            mean_stddev = get_mean_stddev(X)
            X_norm = z_score_normalize(X, mean_stddev)
            reg = LinearRegression().fit(X_norm, Y)
            R2_scores[e] = reg.score(X_norm, Y)
            coefficients[e] = reg.coef_
            del reg
        R2.append(R2_scores)
        COEFF.append(coefficients)
    R2_df, COEFF_df = get_r2_and_coeffs(R2, COEFF, models_names, features_list)
    plot_r2(R2_df, dataset_name)
    plot_coeffs(COEFF_df, dataset_name)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Managerie plot on csv file')
        parser.add_argument('--jsonsf', metavar='path', required=True,
                            help='path to images')
        parser.add_argument('--results_list', metavar='path', required=True,
                            help='txt file containing results files paths')
        parser.add_argument('--csv', metavar='path', required=True,
                            help='path to csv file listing all pairs')
        parser.add_argument('--suffix', metavar='path', required=False,
                            default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                            )
        parser.add_argument('--iterations', metavar='path', required=False, default=100,
                            help='iterations for samples picking')
        parser.add_argument('--samples', metavar='path', required=False, default=200,
                            help='number of samples to pick at each iteration')
        args = parser.parse_args()
        if args.jsonsf == "" or args.results == "" or args.csv == "":
            print("Please, check your input parameters")
    except Exception as e:
        print(e)
        sys.exit(1)
    with open(args.jsonsf, 'r') as fp:
        json_file = json.load(fp)
    keys = list(json_file[list(json_file.keys())[0]].keys())
    selected_measures = ['FA', 'FR']
    errors_list = []
    model_data = []
    models_names = [x.strip().split("/")[-1].split("_")[0] for x in args.results_list]
    for mr in args.results_list:
        errors_list.append(counter(json_file, mr, args.csv, keys))
    for el in errors_list:
        model_data.append(get_data_means(el, keys))
    for selected_measure in selected_measures:
        plot_correlations(model_data, models_names, selected_measure, args.dataset_name)
        plot_feature_weights(model_data, models_names, selected_measure, int(args.iterations), int(args.n_samples),
                             args.dataset_name)
