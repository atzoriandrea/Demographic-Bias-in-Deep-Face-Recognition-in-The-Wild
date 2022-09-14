import datetime
import sys
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../../')
import torch
from torch.utils.data import DataLoader
from data_processor.train_dataset import ImageDataset
import gc
import numpy as np
import os


def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".pt"):
                paths.append(os.path.join(root, file))
    return paths


def load_model(model_path):
    model = torch.load(model_path)
    try:
        while model.module.module != None:
            model = model.module
    except:
        pass
    model.eval()
    device = torch.device('cuda:0')
    return model, device


def csv_to_test(csv_file_name, basepath, bs):
    def compute_correct_batch_size(group_list):
        max = 0
        l = len(group_list)
        for bs in range(1, 64):
            if l % bs == 0:
                max = bs
        return max

    ref = ".diveface_ref.txt"
    cmp = ".diveface_cmp.txt"
    group_ref, group_cmp, ref_id = [], [], []
    with open(csv_file_name, "r") as csv_file:
        csv = csv_file.readlines()
        with open(ref, "w") as ref_file:
            with open(cmp, "w") as cmp_file:
                for line in csv:
                    data = line.strip().split(",")
                    try:
                        data.remove('')
                    except:
                        pass
                    r = data[0]
                    c = data[1]
                    cls = data[2]
                    ref_id.append(data[3])
                    if len(data) == 6:
                        group_ref.append(int(data[4].strip()))
                        group_cmp.append(int(data[5].strip()))
                    else:
                        group_ref.append(int(data[4].strip()))
                        group_cmp.append(int(data[4].strip()))
                    ref_file.write(" ".join([r, cls, "\n"]))
                    cmp_file.write(" ".join([c, cls, "\n"]))
    bs = compute_correct_batch_size(group_ref) if bs is None else bs
    dl_ref = DataLoader(ImageDataset(basepath, ref), bs, shuffle=False, num_workers=4, drop_last=True)
    dl_cmp = DataLoader(ImageDataset(basepath, cmp), bs, shuffle=False, num_workers=4, drop_last=True)
    size = bs * len(dl_ref)
    return dl_ref, dl_cmp, group_ref[:size], group_cmp[:size], ref_id[:size]


def inference(dl_ref, dl_cmp, device, model):
    bs = dl_ref.batch_size
    ref = torch.zeros((bs * len(dl_ref), 512))  # .to(device)
    matches = torch.zeros((bs * len(dl_ref),))
    cmp = torch.zeros((bs * len(dl_ref), 512))
    with tqdm(total=2 * len(dl_ref)) as pbar:
        for batch_idx, (images, labels) in enumerate(dl_ref):
            images = images.to(device)
            labels = labels  # .to(device)
            ref[batch_idx * bs: (batch_idx + 1) * bs] = model.module.backbone.forward(images).cpu().detach()
            matches[batch_idx * bs: (batch_idx + 1) * bs] = labels
            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(1)

        for batch_idx, (images, labels) in enumerate(dl_cmp):
            images = images.to(device)
            labels = labels.to(device)
            cmp[batch_idx * bs: (batch_idx + 1) * bs] = model.module.backbone.forward(images).cpu().detach()
            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(1)
    return ref.cpu().detach(), cmp.cpu().detach(), matches

def pipeline(args):
    bp = args.base
    model_needed = False
    with open(args.cmps, "r") as cmps_file:
        comparisons = cmps_file.readlines()
    for comparison in comparisons:
        if comparison.strip().endswith(".csv"):
            model_needed = True
            break
    if model_needed:
        model, device = load_model(args.model)
        model_t = args.model.strip().split("/")[-1]
    for comparison in comparisons:
        comparison = comparison.rstrip("\n")
        model_name = args.model.strip().split("/")[-1]
        directory = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        directory = directory + "-" + model_name + "-" + str(comparison.split("/")[-1])
        os.makedirs(directory)
        if comparison.endswith(".csv"):
            filename = comparison.split("/")[-1].replace(".csv", "")

            print("Preparing test files...")
            dl_ref, dl_cmp, group_ref, group_cmp, ref_id = csv_to_test(comparison, bp, None)
            print("Inferencing dataset...")
            ref, cmp, matches = inference(dl_ref, dl_cmp, device, model)
            m = matches.flatten()
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            results = cos(ref, cmp)
            np_res = np.vstack([results.numpy(), m.numpy(), group_ref, group_cmp, list(map(int, ref_id))]).T
            outcos = os.path.join(directory, "_".join([model_t, "_results_", filename, ".npy"]))
            np.save(outcos, np_res)


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--model', metavar='path',
                            help='path to model')
        parser.add_argument('--models', metavar='path',
                            help='path to model(s)')
        parser.add_argument('--cmps', metavar='path', required=True,
                            help='path to list of csv files')
        parser.add_argument('--base', metavar='path',
                            help='path to dataset basepath')
        args = parser.parse_args()
        if args.models is None and args.model is None:
            raise Exception
    except Exception as e:
        print(e)
        sys.exit(1)
    if args.model is not None:
        pipeline(args)
    else:
        if args.models.endswith(".txt"):
            models = open(args.models, "r").readlines()
        else:
            models = get_files_full_path(args.models)
        for model in models:
            args.model = model.strip()
            pipeline(args)