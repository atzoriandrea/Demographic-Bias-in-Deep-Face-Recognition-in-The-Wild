import argparse
import os
import sys

from get_data_means import *
import datetime


def get_latex_table(args, save=False):
    folder_name = "Latex_tables"
    with open(args.jsonsf, 'r') as fp:
        json_file = json.load(fp)
    keys = list(json_file[list(json_file.keys())[0]].keys())
    errors = counter(json_file, args.results, args.csv, keys)
    df = get_data_means(errors, keys)
    df_FA = df[['genre', 'ethnicity', 'FA']].groupby(["ethnicity", 'genre']).mean()
    df_FA['FA'] = df_FA['FA'] / 50
    df_FA = df_FA.pivot_table(index='ethnicity', columns='genre')
    df_FA.loc['A+B+C'] = df_FA.mean(numeric_only=True, axis=0)
    df_FA.loc[:, ('FA', 'M+F')] = df_FA.mean(numeric_only=True, axis=1)
    df_FR = df[['genre', 'ethnicity', 'FR']].groupby(["ethnicity", 'genre']).mean()
    df_FR['FR'] = df_FR['FR'] / 5
    df_FR = df_FR.pivot_table(index='ethnicity', columns='genre')
    df_FR.loc['A+B+C'] = df_FR.mean(numeric_only=True, axis=0)
    df_FR.loc[:, ('FR', 'M+F')] = df_FR.mean(numeric_only=True, axis=1)
    df_FA_FR = df_FA.join(df_FR, on='ethnicity')

    txt = "resizebox{\columnwidth}{!}{%\n" + df_FA_FR.to_latex() + "%\n}"

    if save:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with open(os.path.join(folder_name, "table" + dt + ".txt"), "w") as f:
            f.write(txt)
    else:
        print(txt)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--jsonsf', metavar='path', required=True,
                            help='path to images')
        parser.add_argument('--results', metavar='path', required=True,
                            help='path to .npy results file')
        parser.add_argument('--csv', metavar='path', required=True,
                            help='path to csv file listing all pairs')
        parser.add_argument('--save', metavar='path', required=False,
                            help='path to csv file listing all pairs')
        args = parser.parse_args()
        if args.jsonsf == "" or args.results == "" or args.csv == "":
            print("Please, check your input parameters")
    except Exception as e:
        print(e)
        sys.exit(1)
    get_latex_table(args)
