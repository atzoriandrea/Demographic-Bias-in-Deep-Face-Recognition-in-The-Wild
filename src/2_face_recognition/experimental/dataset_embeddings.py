import gc
import os
import sys

from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
import numpy as np
from data_processor.train_dataset import ImageDataset
import torch
from torch.utils.data import DataLoader


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


def compute_correct_batch_size(group_list):
    max = 0
    l = len(group_list)
    for bs in range(1, 48):
        if l % bs == 0:
            max = bs
    return max


def inference(dataloader, model_path):
    model, device = load_model(model_path)
    bs = dataloader.batch_size
    embeddings = torch.zeros((bs * len(dataloader), 512))  # .to(device)
    identities = torch.zeros((bs * len(dataloader),))  # .to(device)
    with tqdm(total=len(dataloader)) as pbar:
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            embeddings[batch_idx * bs: (batch_idx + 1) * bs] = model.module.backbone.forward(images).cpu().detach()
            identities[batch_idx * bs: (batch_idx + 1) * bs] = labels
            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(1)
    return embeddings.cpu().detach().numpy(), identities.cpu().detach().numpy()


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--models', metavar='path',
                            help='path to model(s)')
        parser.add_argument('--images_list', metavar='path', required=True,
                            help='path to dataset basepath')
        args = parser.parse_args()
        if args.models is None and args.model is None:
            raise Exception
    except Exception as e:
        print(e)
        sys.exit(1)
    if args.models.endswith(".txt"):
        models = open(args.models, "r").readlines()
    else:
        models = get_files_full_path(args.models)
    for model in models:
        model_name = model.strip().split("/")[-1]
        with open(args.images_list, "r") as bpts:
            text_files = bpts.readlines()
        for b in text_files:
            with open(b.strip(), "r") as f:
                file = f.readlines()
            root = "/".join(file[0].strip().split(" ")[0].split("/")[:-2])
            over = "/".join(file[0].strip().split(" ")[0].split("/")[:-3])
            bs = 32#compute_correct_batch_size([_ for _ in range(len(file))])
            print("Inferencing %s with %s" % (root, model_name) )
            if not os.path.exists(root+model_name+".npy"):
                data_loader = DataLoader(ImageDataset(root, b.strip()),
                                               bs, shuffle=False, drop_last=True)
                embeddings, identities = inference(data_loader, model)

                np.save(root+model_name+".npy", np.hstack([embeddings, identities.reshape(-1,1)]))
                gc.collect()
                torch.cuda.empty_cache()