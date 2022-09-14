import sys
import gc
import os.path
from tqdm import tqdm
import torch
from yoon_model import Discriminator
from yoon_data import discriminator_data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_discriminator(model_path):
    w = torch.load(model_path)#("/media/Workspace/Models/JSTSP/model_epoch_009.pth")
    D_h2l = Discriminator(16).cuda()
    D_h2l.load_state_dict(w['D_h2l'])
    D_h2l.eval()
    return D_h2l


def get_scores(dataroot, h2l, noise_params):
    D_h2l = load_discriminator(h2l)
    data = discriminator_data(dataroot, noise_func=noise_params)
    loader = DataLoader(dataset=data, batch_size=1024, shuffle=False)
    with tqdm(total=len(loader)) as pbar:
        for b, sample in enumerate(loader):
            original_temp = sample["original"].numpy()
            original = torch.from_numpy(np.ascontiguousarray(original_temp)).cuda()

            with torch.no_grad():
                if b == 0:
                    o = D_h2l.forward(original)
                    pred_on_ori = o.cpu().numpy()

                else:
                    o = D_h2l.forward(original)
                    pred_on_ori = np.concatenate([pred_on_ori, o.cpu().numpy()], axis=None)

            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(1)

    return pred_on_ori


def get_noises_params():
    gaussian_blur_sigmas = [round(x, 3) for x in np.arange(2, 4.2, 0.2)]
    brightness_betas = [round(x, 3) for x in np.arange(1., 3.5, 0.5)]
    gaussian_noise_sigmas = [round(x, 3) for x in np.arange(10, 50, 10)]
    salt_and_pepper_probs = [round(x, 3) for x in np.arange(0.030, 0.18, 0.03)]
    params = [gaussian_blur_sigmas, brightness_betas, gaussian_noise_sigmas, salt_and_pepper_probs]
    noise_types = ["gaussian_blur", "brightness", "gaussian_noise", "S&P"]
    parameters_dicts = []
    for t, p in zip(noise_types, params):
        for v in p:
            if t=="gaussian_blur" or t=="gaussian_noise":
                parameters_dicts.append({"type" : t, "sigma" : v})
            if t == "brightness":
                parameters_dicts.append({"type": t, "beta": v})
            if t=="S&P":
                parameters_dicts.append({"type": t, "probability": v})

    return parameters_dicts


def plot(array, cmp_type, labels,  out_dir):
    path = os.path.join(out_dir, "_".join([cmp_type, "comparations.png"]))
    if os.path.exists(path):
        return
    qmul = np.load("Discr_on_QMUL.npy")
    tinyFace = np.load("Discr_on_TinyFace.npy")
    num = np.min(np.asarray([array.shape[1], len(qmul), len(tinyFace)]))

    sampled_array = np.zeros((array.shape[0], num))
    for i in range(sampled_array.shape[0]):
        sampled_array[i] = np.random.choice(array[i, :], size=num, replace=False)

    sampled_qmul = np.random.choice(qmul, size=num, replace=False)
    sampled_tinyFace = np.random.choice(tinyFace, size=num, replace=False)

    samples = np.vstack([sampled_array, sampled_qmul, sampled_tinyFace])
    labels.extend(["QMUL (LR)", "TinyFace (LR)"])
    fig = plt.figure(figsize=(20, 20), facecolor='white')
    column_values = labels
    df = pd.DataFrame(data=samples.T,
                      columns=column_values)
    df.plot.density()
    plt.savefig(path)


def get_label(text):

    for symbol in ["[", "]"]:
        text = text.replace(symbol, "")
    noise_type, value = text.split(",")
    if "gaussian_noise" in noise_type:
        return "gn"+value
    if "brightness" in noise_type:
        return "br"+value
    if "gaussian_blur" in noise_type:
        return "gb"+value
    if "S&P" in noise_type:
        return "S&P"+value


def execute(datarootHR, datarootGEN, h2l, out_dir):
    params = get_noises_params()
    rootsHR, rootsGEN = [], []

    if datarootHR.endswith(".txt"):
        with open(datarootHR, "r") as dr:
            rows = dr.readlines()
        for l in rows:
            rootsHR.append(l.strip())
    else:
        rootsHR.append(datarootHR)

    if datarootGEN.endswith(".txt"):
        with open(datarootGEN, "r") as dr:
            rows = dr.readlines()
        for l in rows:
            rootsGEN.append(l.strip())
    else:
        rootsGEN.append(datarootHR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for rootHR, rootGEN in zip(rootsHR, rootsGEN):
        ds_name = rootHR.split("/")[-2]
        sub_dir = os.path.join(out_dir, ds_name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        index = 0
        names = []

        print("Computing %s - %s" % (ds_name, "GENERATED"))
        out_file_gen = os.path.join(sub_dir, "GENERATED")
        if os.path.exists(out_file_gen + ".npy"):
            print("File %s already exists. Skipping inference..." % out_file_gen)
            comparations = np.load(out_file_gen + ".npy")
        else:
            res_generated = get_scores([rootGEN], h2l, None)
            comparations = res_generated
            np.save(out_file_gen + ".npy", res_generated)
        names.append("GAN")

        print("Computing %s - %s" % (ds_name, "HIGHRES"))
        out_file_highres = os.path.join(sub_dir, "HIGHRES")
        if os.path.exists(out_file_highres + ".npy"):
            print("File %s already exists. Skipping inference..." % out_file_highres)
            comparations = np.vstack([comparations, np.load(out_file_highres + ".npy")])
        else:
            res_highres = get_scores([rootHR], h2l, None)
            comparations = np.vstack([comparations, res_highres])
            np.save(out_file_highres + ".npy", res_highres)
        names.append("HR")

        for i, p in enumerate(params):
            try:
                values = list(p.values())
                values[1] = round(values[1], 3)
                name = "".join(str(values))
                cmp_dir = os.path.join(sub_dir, p["type"])
                print("Computing %s - %s" % (ds_name, name))
                out_file = os.path.join(cmp_dir, name)
                if os.path.exists(out_file+".npy"):
                    print("File %s already exists. Skipping inference..." % out_file)
                    comparations = np.vstack([comparations, np.load(out_file+".npy")])
                else:
                    res = get_scores([rootHR], h2l, p)
                    if not os.path.exists(cmp_dir):
                        os.makedirs(cmp_dir)
                    np.save(out_file+".npy", res)
                    comparations = np.vstack([comparations, res])
                names.append(get_label(name))

                if i == len(params)-1 or params[i]["type"] != params[i+1]["type"]:
                    plot(comparations, params[i]["type"], names, sub_dir)
                    comparations = comparations[:2, :]
                    names = names[:2]
            except Exception:
                exc_type, value, traceback = sys.exc_info()
                #assert exc_type.__name__ == 'NameError'
                print("Failed with exception [%s]" % exc_type.__name__)
                print("Failed with %s" % name)


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--datarootHR', metavar='path',
                            help='Dataset path', required=True)
        parser.add_argument('--datarootGEN', metavar='path',
                            help='Dataset path', required=True)
        parser.add_argument('--h2l', metavar='path',
                            help='path to model', required=True)
        parser.add_argument('--out', metavar='path',
                            help='outpath', required=True)
        args = parser.parse_args()
    except Exception as e:
        print(e)
        sys.exit(1)
    execute(args.datarootHR, args.datarootGEN, args.h2l, args.out)