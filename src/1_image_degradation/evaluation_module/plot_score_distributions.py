import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def get_files_full_path(rootdir):
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".npy"):
                paths.append(os.path.join(root, file))
    return paths


def get_scores_dataframe(files_list):
    dfs = []
    for f in files_list:
        df = pd.DataFrame(columns=["Score", "Dataset", "Noise", "Value", "Index"])
        x = np.load(f)
        ds = f.split("/")[4]
        if f.endswith("GENERATED.npy"):
            noise = "GAN-Degraded"
            value = 0
        elif f.endswith("HIGHRES.npy"):
            noise = "High-Resolution"
            value = 0
        else:
            t = f.split("/")[-1].replace("[", "").replace("]", "").replace(".npy", "")
            t2 = t.split(",")
            noise = t2[0]
            value = t2[1]
        df["Score"] = x
        df["Dataset"] = ds
        df["Noise"] = noise
        df["Value"] = value
        df["Index"] = list(range(x.shape[0]))
        dfs.append(df)

    tot = pd.concat(dfs)
    means = tot.groupby(["Noise", "Dataset", "Index"]).mean().reset_index()
    return means

def add_qmul(qmul_results_file, means):
    qmul = np.load(qmul_results_file)
    df = pd.DataFrame(columns=["Score", "Dataset", "Noise", "Index"])
    df["Score"] = qmul
    df["Dataset"] = "QMUL"
    df["Noise"] = "Original LR (QMUL)"
    df["Index"] = list(range(qmul.shape[0]))
    means = pd.concat([means, df]).reset_index()
    return means

def get_plot(means):
    fig, ax = plt.subplots(5, 1, figsize=(10, 20), facecolor='white')
    datasets = ["DiveFace", "MAAD", "CelebA", "RFW", "BUPT"]
    sns.set_style(style="white")
    fig.subplots_adjust(hspace=.4)
    for i, d in enumerate(datasets):
        sub = means[(means["Dataset"] == d) | (means["Dataset"] == "QMUL")].drop(columns=["Index", "index"])
        ax[i].title.set_text(d)
        sns.kdeplot(data=sub, x="Score", hue="Noise", ax=ax[i])
        ax[i].set_facecolor('white')
    plt.savefig("distributions.png", bbox_inches="tight")


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--files_basepath', metavar='path',
                            help='basepath of npy files computed with the get_discriminators_scores script', required=True)
        parser.add_argument('--qmul_results', metavar='path',
                            help='path to qmul npy file computed with the get_discriminators_scores script', required=False)
        args = parser.parse_args()
    except Exception as e:
        print(e)
        sys.exit(1)
    files = get_files_full_path(args.files_basepath)
    scores = get_scores_dataframe(files)
    reference_tot_scores = add_qmul(args.qmul_results, scores)
    get_plot(reference_tot_scores)