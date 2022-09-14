import sys
import argparse
import os
from get_discriminators_scores import get_noises_params, get_label
import numpy as np
import pandas as pd
from dictances import bhattacharyya
from scipy.stats import kruskal
import matplotlib.pyplot as plt


def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".npy"):
                paths.append(os.path.join(root, file))
    return paths


def search_file(files_list, target):
    for f in files_list:
        if target in f:
            return f
    return None


def get_name(dict_values):
    values = list(dict_values.values())
    values[1] = round(values[1], 3)
    name = "".join(str(values))
    return name


def get_table(dataroot):
    datasets = os.listdir(dataroot)
    tests = get_noises_params()
    rows = len(datasets)
    dataframe_columns = [get_label("".join(str(list(x.values())))) for x in tests]
    dataframe_columns.append("GENERATED")
    results = np.zeros((rows, len(dataframe_columns)))
    tests.append("GENERATED")
    reference = np.load("Discr_on_QMUL.npy")
    for i, d in enumerate(datasets):
        data_files = sorted(get_files_full_path(os.path.join(dataroot, d)))
        for j, p in enumerate(tests):
            name = get_name(p) if p != "GENERATED" else "GENERATED"
            target = search_file(data_files, name)
            data = np.load(target)
            print(" - ".join([d, name]))
            num_samples = min(data.shape[0], reference.shape[0])
            sampled_data = np.random.choice(data, size=num_samples, replace=False)
            sampled_reference = np.random.choice(reference, size=num_samples, replace=False)
            stat, p = kruskal(sampled_data, sampled_reference)
            print('stat=%.3f, p=%f' % (stat, p))
            results[i, j] = np.format_float_scientific(round(stat, 2), exp_digits=4)

    df = pd.DataFrame(data=results,
                      index=datasets,
                      columns=dataframe_columns)
    df = df.apply( lambda series: series.apply (lambda x : format(x,'.1E')))
    with open("latex.txt", "w") as cmps:
        cmps.write(df.T.to_latex())

    fig = plt.figure()
    c = df.to_latex().replace('\n', ' ')
    plt.text(9, 3.4, c, size=12)
    plt.savefig("table.png")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--dataroot', metavar='path',
                            help='Data path', required=True)
        args = parser.parse_args()
    except Exception as e:
        print(e)
        sys.exit(1)
    get_table(args.dataroot)
