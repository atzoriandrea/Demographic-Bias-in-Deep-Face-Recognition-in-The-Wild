import parser
import sys
from identification_files_dict import *
from tqdm import tqdm
import numpy as np
from ordered_set import OrderedSet
import torch.nn as nn
import torch
import random
import json
import gc
import os


def get_idx_full(v):
    if v["genre"] == "Male":
        pred = [0, 2, 4]
    else:
        pred = [1, 3, 5]
    if v["ethnicity"] == "Asian":
        pred = pred[0]
    elif v["ethnicity"] == "Black":
        pred = pred[1]
    else:
        pred = pred[2]
    return pred


def get_idx_eth(v):
    if v["ethnicity"] == "Asian":
        pred = 0
    elif v["ethnicity"] == "Black":
        pred = 1
    else:
        pred = 2
    return pred


def get_idx_genre(v):
    if v["genre"] == "Male":
        pred = 0
    else:
        pred = 1
    return pred


def get_idx_fun(dataset_name):
    if dataset_name in ["MAAD", "CelabA"]:
        return get_idx_genre
    elif dataset_name in ["BUPT", "RFW"]:
        return get_idx_eth
    else:
        return get_idx_full


def identification_analisys(cmp_type, npy_file, json_file, test_set_file, get_idx, basepath, isBUPT=False):
    datafile = np.load(npy_file)

    with open(json_file, "r") as fp:
        features = json.load(fp)
    with open(test_set_file, "r") as ts:
        test_set = ts.readlines()

    device = torch.device("cuda:0")

    classes_json = {}
    for k, v in features.items():
        p = v["path"].split("/")[-2]
        classes_json[p] = get_idx(v)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    groups = []
    for i in test_set:
        p = i.strip().split(" ")[0].split("/")[-2]
        g = classes_json[p]
        groups.append(g)
    groups = np.asarray(groups)[:datafile.shape[0]]

    groups_unique = np.unique(groups)

    sample_idx = 0
    sample_step = 10000
    if isBUPT:
        n_gr = groups_unique.shape[0]
        sampled_data = np.zeros((n_gr * sample_step, 513))
        sampled_groups = np.zeros((n_gr * sample_step,))
        for g in groups_unique:
            datafile_sub = datafile[groups == g]
            np.random.seed(41)
            sample_indexes = np.random.choice(range(datafile_sub.shape[0]), sample_step, replace=False)
            sampled_data[sample_idx * sample_step: (sample_idx + 1) * sample_step] = datafile_sub[sample_indexes]
            sampled_groups[sample_idx * sample_step: (sample_idx + 1) * sample_step] = groups[groups == g][sample_indexes]
            sample_idx += 1
        datafile = sampled_data
        groups = sampled_groups


    tot_size = 0
    for g in groups_unique:
        datafile_sub = datafile[groups == g]
        identities = OrderedSet(datafile_sub[:, -1])
        tot_size += len(identities) * (datafile_sub.shape[0] - 1)

    cosines = torch.zeros((tot_size,))
    matches = torch.zeros((tot_size,))
    group_ref = torch.zeros((tot_size,))
    group_cmp = torch.zeros((tot_size,))
    ref_id = torch.zeros((tot_size,))

    curr_idx = 0
    last_curr_idx = 0

    for g in groups_unique:
        datafile_sub = datafile[groups == g]
        identities = OrderedSet(datafile_sub[:, -1])

        gpu_fit = False
        splits = 1

        datafile_torch = torch.tensor(datafile_sub, dtype=torch.float16).to(device)

        while not gpu_fit:
            size = (np.ceil(len(identities) / splits) * (datafile_sub.shape[0] - 1))
            try:
                gpu_cosines = torch.zeros((int(size),)).to(device)
                gpu_fit = True
                del gpu_cosines
                gc.collect()
                torch.cuda.empty_cache()
            except:
                gc.collect()
                torch.cuda.empty_cache()
                splits = splits * 4 if splits <= 4 else splits * 2
        print("Array needed %d splits " % splits)

        gpu_curr_idx = 0
        last_gpu_size = 0
        splitted_identities = [identities] if splits == 1 else [list(x) for x in np.array_split(identities, splits)]

        with tqdm(total=len(identities)) as pbar:
            # assert sum([len(ids) * (datafile_torch.shape[0] - 1) for ids in splitted_identities]) == cosines.shape[0]
            for s, ids in enumerate(splitted_identities):
                gpu_cosines = torch.zeros((len(ids) * (datafile_torch.shape[0] - 1),)).to(device)
                assert gpu_cosines.shape[0] - last_gpu_size == sum(
                    [len(ids) * (datafile_torch.shape[0] - 1) for ids in splitted_identities[s:]])
                for index, id in enumerate(ids):
                    mated = datafile_sub[datafile_sub[:, -1] == id][:, :-1]
                    probe_idx = random.randint(0, len(mated) - 1)
                    probe = torch.tensor(mated[probe_idx]).to(device)
                    mates2 = torch.tensor(np.delete(mated, probe_idx, axis=0)).to(device)

                    unmated = datafile_torch[datafile_sub[:, -1] != id][:, :-1]
                    output = cos(probe, torch.vstack([mates2, unmated]))
                    m = torch.hstack([torch.ones((mates2.shape[0],)),
                                      torch.zeros((unmated.shape[0]))])

                    gpu_cosines[gpu_curr_idx: gpu_curr_idx + output.shape[0]] = output
                    matches[curr_idx: curr_idx + output.shape[0]] = m
                    group_ref[curr_idx: curr_idx + output.shape[0]] = torch.full((output.shape[0],),
                                                                                 g)  # torch.tensor(groups[x[:,-1] == id])[:-1]
                    group_cmp[curr_idx: curr_idx + output.shape[0]] = torch.full((output.shape[0],), g)
                    ref_id[curr_idx: curr_idx + output.shape[0]] = torch.full((output.shape[0],), id)
                    curr_idx += output.shape[0]
                    gpu_curr_idx += output.shape[0]
                    pbar.update(1)
                    if id == ids[-1]:
                        gpu_curr_idx = 0
                        cosines[last_curr_idx:curr_idx] = gpu_cosines.cpu().detach()
                        last_curr_idx = curr_idx
                        last_gpu_size += gpu_cosines.shape[0]
                del gpu_cosines, mates2, probe, unmated
                gc.collect()
                torch.cuda.empty_cache()
    if curr_idx == cosines.shape[0]:
        print("Size was correct")
    else:
        print("Size was not correct")

    inferenced = torch.hstack([
        cosines.reshape(-1, 1),
        matches.reshape(-1, 1),
        group_ref.reshape(-1, 1),
        group_cmp.reshape(-1, 1),
        ref_id.reshape(-1, 1)
    ])

    savepath = os.path.join(basepath, cmp_type + ".npy")
    print("Saving %s... " % savepath)
    np.save(savepath, inferenced.numpy())

    return 0


if __name__ == '__main__':
    for dataset_name, comparisons in data.items():
        if comparisons is not None:
            dataset_path = os.path.join("/media/Workspace/JSTSP/RQ2_B", model_name, dataset_name)
            grp_fun = get_idx_fun(dataset_name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            for cmp_type, cmp_files in comparisons.items():
                print("Inferencing %s - %s - %s" % (model_name, dataset_name, cmp_type))
                if not os.path.exists(os.path.join(dataset_path, cmp_type + ".npy")):
                    identification_analisys(cmp_type,
                                            cmp_files["npy"],
                                            cmp_files["json"],
                                            cmp_files["testfile"],
                                            grp_fun,
                                            dataset_path,
                                            dataset_name in ["BUPT", "CelabA"])
