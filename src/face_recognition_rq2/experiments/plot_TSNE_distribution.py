import argparse
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
import pandas as pd
from sklearn.manifold import TSNE
#from experimental_menagerie_predictor_trainer import prepare_data


def prepare_data():
    emb_fs1 = pd.DataFrame(np.hstack(
        [TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data_fs1_norm), quadrant, fars]),
                           columns=['x', 'y', 'label', 'FAR'])
    emb_fs2 = pd.DataFrame(np.hstack(
        [TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data_fs2_norm), quadrant, fars]),
                           columns=['x', 'y', 'label', 'FAR'])


def plot(title, emb_fs1, emb_fs2):
    fig, ax = plt.subplots(2,4, figsize=(40,20))
    embs = [emb_fs1, emb_fs2]
    titles = ["9-Quad", "Genuines", "Imposters", "FARs > mean"]
    for i in range(8):
        c = i%4
        r = int(i/4)
        d = embs[r]
        print(np.unique(d['label']))
        if c == 0:
            ax[r,c].set_title(titles[c])
            for k,d in d.groupby('label'):
                ax[r,c].scatter(d['x'], d['y'], label=int(k))
        if c == 1:
            v = d['label'].apply(lambda x : x % 3)
            ax[r,c].set_title(titles[c])
            d_temp = copy.deepcopy(d)
            d_temp['label'] = d_temp['label'].apply(lambda x : int((x-1) % 3)+1)
            for k,d in d_temp.groupby('label'):
                ax[r,c].scatter(d['x'], d['y'], label=k)
        if c == 2:
            v = d['label'].apply(lambda x : int(x / 3))
            ax[r,c].set_title(titles[c])
            d_temp = copy.deepcopy(d)
            d_temp['label'] = d_temp['label'].apply(lambda x : ((x-1)//3)+1)
            for k,d in d_temp.groupby('label'):
                ax[r,c].scatter(d['x'], d['y'], label=k)
        if c == 3:

            ax[r,c].set_title(titles[c])
            mean = d['FAR'].mean()
            v = d['FAR'].apply(lambda x : x > mean)
            d_temp = copy.deepcopy(d)
            d_temp['FAR'] = d_temp['FAR'].apply(lambda x : "FAR > mean" if x > mean else "FAR <= mean")
            for k,d in d_temp.groupby('FAR'):
                ax[r,c].scatter(d['x'], d['y'], label=k)
        ax[r,c].legend()
        #ax[r,c].legend(handles=handles,loc='best', frameon=False)
    plt.savefig("Quadrant Plots.png",facecolor='white')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Managerie plot on csv file')
        parser.add_argument('--jsonsf', metavar='path', required=True,
                            help='path to images')
        parser.add_argument('--results', metavar='path', required=True,
                            help='')
        parser.add_argument('--csv', metavar='path', required=True,
                            help='')
        args = parser.parse_args()
        if args.jsonsf == "" or args.results == "" or args.csv == "":
            print("Please, check your input parameters")
    except Exception as e:
        print(e)
        sys.exit(1)
    dm, negatives, positives = data_means(json_file=args.jsonsf, results_file=args.results, couples_csv_file=args.csv)
    title = args.csv.split(".")[-2].split("/")[-1]
    title = "managerie_"+title
    plot(dm, negatives, positives, title)