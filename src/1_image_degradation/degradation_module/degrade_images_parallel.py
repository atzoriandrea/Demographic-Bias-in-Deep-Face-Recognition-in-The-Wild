import gc
import os.path
import sys

import torch
from tqdm import tqdm

from src.image_degradation.training_module.model import High2Low
from src.image_degradation.data import high2low_data
from torch.utils.data import DataLoader
import numpy as np
import cv2

Test_Data = ["/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/Dataset/7259735@N07_identity_10",
             "/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/Dataset/7597392@N03_identity_7",
             "/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/Dataset/8495919@N02_identity_31"]


def load_model(model_path):
    w = torch.load(model_path)#("/media/Workspace/Models/JSTSP/model_epoch_009.pth")
    G_h2l = High2Low().cuda()
    #D_h2l = Discriminator(16).cuda()
    G_h2l.load_state_dict(w['G_h2l'])
    #D_h2l.load_state_dict(w['D_h2l'])
    G_h2l.eval()
    return G_h2l
    #D_h2l.eval()


def find_folder_idx(obj, l):
    for i, el in l[::-1]:
        if el == obj:
            return i+1
    return None


def degrade(data_root, model_path):
    G_h2l = load_model(model_path)
    data = high2low_data([data_root])
    loader = DataLoader(dataset=data, batch_size=1024, shuffle=False)
    path = data_root.split("/")
    expath = path[-1]
    path[-1] = "Wild"+path[-1]
    dest_path = "/".join(path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with tqdm(total=len(loader)) as pbar:
        for b, sample in enumerate(loader):
            zs = sample["z"].cuda()
            high_temp = sample["hr"].numpy()
            paths = sample["path"]
            high = torch.from_numpy(np.ascontiguousarray(high_temp)).cuda()
            with torch.no_grad():
                low_gen = G_h2l(high, zs)
            for i, lg in enumerate(low_gen):
                np_gen = lg.detach().cpu().numpy().transpose(1, 2, 0)
                np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
                np_gen = (np_gen * 255).astype(np.uint8)
                folder = os.path.join(dest_path, paths[i].split("/")[-2])
                image_name = paths[i].split("/")[-1]
                if not os.path.exists(folder):
                    os.makedirs(folder)
                new_path = os.path.join(folder, image_name)
                cv2.imwrite(new_path, cv2.resize(np_gen, (112,112)))
            #print("DONE: batch %d of %d" % (b+1, len(loader)))
            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(1)
#test_save = "degraded"


#test_data = high2low_data(Test_Data)
#test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--dataroot', metavar='path',
                            help='Dataset path', required=True)
        parser.add_argument('--h2l', metavar='path',
                            help='path to model', required=True)
        args = parser.parse_args()
    except Exception as e:
        print(e)
        sys.exit(1)
    if args.dataroot.endswith(".txt"):

        with open(args.dataroot, "r") as roots:
            rows = roots.readlines()
        for root in rows:
            degrade(root.strip(), args.h2l)
    else:
        degrade(args.dataroot, args.h2l)
