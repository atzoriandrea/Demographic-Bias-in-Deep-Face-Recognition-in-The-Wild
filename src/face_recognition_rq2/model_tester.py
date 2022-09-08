import datetime
import sys

sys.path.append('../')
sys.path.append('../../')
from training.train import FaceModel
from models_definitions.backbone.backbone_def import BackboneFactory
from models_definitions.head.head_def import HeadFactory
import torch
from torch.utils.data import DataLoader
from data_processor.train_dataset import ImageDataset
import gc
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import os


def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".pt"):
                paths.append(os.path.join(root, file))
    return paths


def csv_to_test(csv_file_name, basepath):
    def compute_correct_batch_size(group_list):
        max = 0
        l = len(group_list)
        for bs in range(1, 17):
            if l % bs == 0:
                max = bs
        return max

    ref = ".ref.txt"
    cmp = ".cmp.txt"
    group_ref, group_cmp, ref_id = [], [], []
    with open(csv_file_name, "r") as csv_file:
        csv = csv_file.readlines()
        with open(ref, "w") as ref_file:
            with open(cmp, "w") as cmp_file:
                for line in csv:
                    data = line.split(",")
                    r = data[0]
                    c = data[1]
                    cls = data[2]
                    ref_id.append(data[3])
                    if len(data) == 6:
                        group_ref.append(int(data[4]))
                        group_cmp.append(int(data[5]))
                    else:
                        group_ref.append(int(data[4]))
                        group_cmp.append(int(data[4]))
                    ref_file.write(" ".join([r, cls, "\n"]))
                    cmp_file.write(" ".join([c, cls, "\n"]))
    bs = compute_correct_batch_size(group_ref)
    dl_ref = DataLoader(ImageDataset(basepath, ref), bs, shuffle=False, num_workers=4)
    dl_cmp = DataLoader(ImageDataset(basepath, cmp), bs, shuffle=False, num_workers=4)
    return dl_ref, dl_cmp, group_ref, group_cmp, ref_id


def inference(dl_ref, dl_cmp, device, model):
    for batch_idx, (images, labels) in enumerate(dl_ref):
        images = images.to(device)
        labels = labels.to(device)
        outputs_temp = model.module.backbone.forward(images)
        if batch_idx == 0:
            ref = outputs_temp.cpu().detach()
            matches = labels.cpu().detach()
        else:
            ref = torch.vstack((ref, outputs_temp.cpu().detach()))
            matches = torch.vstack((matches, labels.cpu().detach()))
        gc.collect()
        torch.cuda.empty_cache()

    for batch_idx, (images, labels) in enumerate(dl_cmp):
        images = images.to(device)
        labels = labels.to(device)
        outputs_temp = model.module.backbone.forward(images)
        if batch_idx == 0:
            cmp = outputs_temp.cpu().detach()
        else:
            cmp = torch.vstack((cmp, outputs_temp.cpu().detach()))
        gc.collect()
        torch.cuda.empty_cache()
    return ref, cmp, matches


def load_model(model_path):
    model = torch.load(model_path)
    try:
        while model.module.module != None:
            model = model.module
    except:
        pass
    model.eval()
    try:
        device = torch.device('cuda:0')
    except:
        device = torch.device('cpu')
    return model, device


def NormalizeData(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def most_sim_idx(far, frr, eer):
    curr_idx = 0
    diff = np.inf
    for el1, el2 in zip(far, frr):
        avg_err = np.average((np.abs(el1 - eer), np.abs(el2 - eer)))
        if avg_err <= diff:
            diff = avg_err
            curr_idx += 1
    return curr_idx


def far1perc(far):
    absolute_val_array = np.abs(far - 0.01)
    smallest_difference_index = absolute_val_array.argmin()
    return smallest_difference_index


def get_far_frr_by_user(user_data, threshold):
    user_data = user_data.astype(np.float32)
    threshold = threshold.astype(np.float32)
    ta = np.sum((user_data[:, 0] >= threshold) & (user_data[:, 2] == 1))
    fr = np.sum((user_data[:, 0] < threshold) & (user_data[:, 2] == 1))
    fa = np.sum((user_data[:, 0] >= threshold) & (user_data[:, 2] == 0))
    tr = np.sum((user_data[:, 0] < threshold) & (user_data[:, 2] == 0))
    user_far = fa / (fa + ta) if fa + ta > 0 else 0
    user_frr = fr / (fr + tr) if fr + tr > 0 else 0
    if np.isnan(user_frr).any() or np.isnan(user_far).any():
        print("NAN!")
    return user_far, user_frr


def get_graphs_and_matrices(cosines, group_ref, m, path):
    groups = list(set(group_ref))
    l = len(groups)
    labels = ["AM", "AW", "BM", "BW", "CM", "CW"]
    colors = ["red", "orange", "blue", "yellow", "violet", "black"]
    rows = l // 3
    cols = l // rows
    fig1, axs1 = plt.subplots(rows, cols, facecolor='white')
    fig1.set_size_inches((15, 10))
    fig2, axs2 = plt.subplots(2, 2, facecolor='white')
    fig2.set_size_inches((15, 15))
    eer_auc_far_frr_far1_frr1 = np.zeros((6, l))
    for grp_idx, group in enumerate(groups):
        subset = cosines[np.asarray(group_ref) == group]
        subset_norm = NormalizeData(subset)
        fpr, tpr, thresholds = metrics.roc_curve(m[np.asarray(group_ref) == group], subset_norm, pos_label=1)
        idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
        if len(idxs) > 0:
            fpr = np.delete(fpr, idxs)
            tpr = np.delete(tpr, idxs)
            thresholds = np.delete(thresholds, idxs)
        fnr = 1 - tpr
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        except:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
        idx_far_frr = most_sim_idx(fpr, fnr, eer)
        idx_f1 = far1perc(fpr)
        thresh = interp1d(fpr, thresholds)(eer)
        axis = axs1[round((group / l) + 0.0001), group % 3] if rows > 1 else axs1[group % 3]
        axis.set_title(" - ".join(['Equal Error Rate', labels[group]]))
        axis.plot(np.sort(fpr)[::-1], label="FAR")  # Sorted in ascending order
        axis.plot(np.sort(fnr), label="FRR")
        axis.axvline(x=(np.abs(thresholds[::-1] - thresh)).argmin(), label='EER = %0.3f' % eer, color="red")
        axis.set_ylabel('Error')
        axis.set_xlabel('Security Threshold')
        step = int(len(thresholds) / 5)
        axis.set_xticks(list(range(len(thresholds)))[::step], np.round(thresholds[::-step], 2))
        axis.set_box_aspect(1.0)
        axis.legend(loc="upper center")
        ##############################################
        roc_auc = metrics.auc(fpr, tpr)
        for row in range(2):
            axs2[row, 0].set_title('Receiver Operating Characteristic')
            axs2[row, 0].plot(fpr, tpr, colors[group], label='AUC ' + " - " + labels[group] + ' = %0.2f' % roc_auc)
            axs2[row, 0].legend(loc='lower right')
            axs2[row, 0].plot([0, 1], [0, 1], 'r--')
            axs2[row, 0].set_xlim([0, 1])
            axs2[row, 0].set_ylim([0, 1])
            axs2[row, 0].set_box_aspect(1.0)
            axs2[row, 0].set_ylabel('True Positive Rate')
            axs2[row, 0].set_xlabel('False Positive Rate')
        axs2[1, 0].set_xlim(0, 0.5)
        axs2[1, 0].set_ylim(0.77, 1)

        ##############################################
        v = ["Negatives", "Positives"]
        for row in range(2):
            eq = subset_norm[m[np.asarray(group_ref) == group] == 1]
            neq = subset_norm[m[np.asarray(group_ref) == group] == 0]
            # res = np.vstack((eq,neq)).transpose()
            axs2[row, 1].set_title(" - ".join(['Cosines Distribution', v[row]]))
            axs2[row, 1].set_box_aspect(1.0)
            sns.kdeplot(ax=axs2[row, 1], data=neq if row == 0 else eq, label=labels[group])
            axs2[row, 1].legend(loc='best')
            axs2[row, 1].set_xlim(0, 1.4)
            axs2[row, 1].set_ylim(0, 7)
        eer_auc_far_frr_far1_frr1[:, grp_idx] = eer, roc_auc, fpr[idx_far_frr], fnr[idx_far_frr], fpr[idx_f1], fnr[
            idx_f1]
        del subset, subset_norm
    fig1.savefig(os.path.join(path, 'FAR_FRR_ERR.png'), dpi=fig1.dpi)
    fig2.savefig(os.path.join(path, 'AUC_CosDensities.png'), dpi=fig2.dpi)
    plt.close('all')
    return eer_auc_far_frr_far1_frr1, groups


def get_heatmaps(eer_auc_far_frr_far1_frr1, path, groups):
    eer, auc, far, frr, far1, frr1 = eer_auc_far_frr_far1_frr1
    labels_tot = ["AM", "AW", "BM", "BW", "CM", "CW"]
    labels = [labels_tot[i] for i in groups]
    eerdiff = np.round(np.array(eer)[:, None] - np.array(eer)[None, :], 3)
    aucdiff = np.round(np.array(auc)[:, None] - np.array(auc)[None, :], 3)
    far_diff = np.round(np.array(far)[:, None] - np.array(far)[None, :], 3)
    frr_diff = np.round(np.array(frr)[:, None] - np.array(frr)[None, :], 3)
    far1_diff = np.round(np.array(far1)[:, None] - np.array(far1)[None, :], 3)
    frr1_diff = np.round(np.array(frr1)[:, None] - np.array(frr1)[None, :], 3)
    titles = ["EER Differences", "AUC Differences", "FAR Differences", "FRR Differences", "FAR (far 1%) Diff",
              "FRR (far 1%) Diff"]
    data = [eerdiff, aucdiff, far_diff, frr_diff, far1_diff, frr1_diff]
    rows, cols = 3, 2
    d = 0
    fig, axs = plt.subplots(rows, cols, facecolor='white')
    fig.set_size_inches((15, 15))
    for r in range(rows):
        for c in range(cols):
            axs[r, c].set_title(titles[d])
            m = axs[r, c].matshow(data[d])
            axs[r, c].set_box_aspect(1.0)
            axs[r, c].set_xticks(list(range(len(labels))), labels)
            axs[r, c].set_yticks(list(range(len(labels))), labels)
            w, h = data[d].shape
            fig.colorbar(m, ax=axs[r, c])
            for i in range(w):
                for j in range(h):
                    t = data[d][i, j]
                    axs[r, c].text(i, j, str(t), va='center', ha='center')
            d += 1
    fig.savefig(os.path.join(path, 'heatmaps.png'), dpi=fig.dpi)
    plt.close('all')


def get_heatmaps_by_id_and_group(cosines, group_ref, ref_id, m, path):
    groups = list(set(group_ref))
    l = len(groups)
    identities = list(set(ref_id))
    avg_fars = np.zeros((len(identities), 6), dtype=np.float32)
    avg_frrs = np.zeros((len(identities), 6), dtype=np.float32)
    labels_tot = ["AM", "AW", "BM", "BW", "CM", "CW"]
    labels = [labels_tot[i] for i in groups]
    last = 0
    last_id = 0
    fprTOT, tprTOT, thresholdsTOT = metrics.roc_curve(m, NormalizeData(cosines), pos_label=1)
    idxsTOT = np.sort(np.where(thresholdsTOT > 1)).flatten()[::-1]
    if len(idxsTOT) > 0:
        fprTOT = np.delete(fprTOT, idxsTOT)
        tprTOT = np.delete(tprTOT, idxsTOT)
        thresholdsTOT = np.delete(thresholdsTOT, idxsTOT)
    fnrTOT = 1 - tprTOT
    eerTOT = brentq(lambda x: 1. - x - interp1d(fprTOT, tprTOT)(x), 0., 1.)
    idx_far_frrTOT = most_sim_idx(fprTOT, fnrTOT, eerTOT)
    idx_f1TOT = far1perc(fprTOT)
    threshTOT = thresholdsTOT[idx_far_frrTOT]  # interp1d(fpr, thresholds)(eer)
    thresh1percTOT = thresholdsTOT[idx_f1TOT]
    for group in groups:
        subset = cosines[np.asarray(group_ref) == group]
        subset_norm = NormalizeData(subset)
        fpr, tpr, thresholds = metrics.roc_curve(m[np.asarray(group_ref) == group], subset_norm, pos_label=1)
        idxs = np.sort(np.where(thresholds > 1)).flatten()[::-1]
        if len(idxs) > 0:
            fpr = np.delete(fpr, idxs)
            tpr = np.delete(tpr, idxs)
            thresholds = np.delete(thresholds, idxs)
        fnr = 1 - tpr
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        except:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
        idx_far_frr = most_sim_idx(fpr, fnr, eer)
        idx_f1 = far1perc(fpr)
        thresh = thresholds[idx_far_frr]  # interp1d(fpr, thresholds)(eer)
        thresh1perc = thresholds[idx_f1]
        ids_sub = ref_id[last:last + len(subset_norm)]
        last += len(subset_norm)
        subset_norm_ids = np.vstack([subset_norm, ids_sub, m[np.asarray(group_ref) == group]]).T
        identities_sub = list(set(ids_sub))
        # avg_fars = np.zeros((len(identities_sub),2))
        # avg_frrs = np.zeros((len(identities_sub),2))
        for p, id in enumerate(identities_sub):
            user_cosines = subset_norm_ids[subset_norm_ids[:, 1] == id]
            user_far, user_frr = get_far_frr_by_user(user_cosines, thresh)
            user_far1perc, user_frr1perc = get_far_frr_by_user(user_cosines, thresh1perc)
            user_farTOT, user_frrTOT = get_far_frr_by_user(user_cosines, threshTOT)
            user_far1percTOT, user_frr1percTOT = get_far_frr_by_user(user_cosines, thresh1percTOT)
            avg_fars[last_id + p] = user_far, user_far1perc, user_farTOT, user_far1percTOT, id, group
            avg_frrs[last_id + p] = user_frr, user_frr1perc, user_frrTOT, user_frr1percTOT, id, group
        last_id += p
    ###########################
    ## INSTANCING HEATMAPS ####
    rows, cols = 4, 2
    fig, axs = plt.subplots(rows, cols, facecolor='white')
    fig.set_size_inches((15, 20))
    ###########################
    group_fars_mean = np.zeros((len(groups), 4))
    group_frrs_mean = np.zeros((len(groups), 4))
    for i, group in enumerate(groups):
        group_fars = avg_fars[avg_fars[:, 5] == group]
        group_frrs = avg_frrs[avg_frrs[:, 5] == group]
        group_fars_mean[i] = np.mean(group_fars[:, :4], axis=0)
        group_frrs_mean[i] = np.mean(group_frrs[:, :4], axis=0)
    fars_diff = np.round(np.array(group_fars_mean[:, 0])[:, None] - np.array(group_fars_mean[:, 0])[None, :], 3)
    frrs_diff = np.round(np.array(group_frrs_mean[:, 0])[:, None] - np.array(group_frrs_mean[:, 0])[None, :], 3)
    fars_1perc_diff = np.round(np.array(group_fars_mean[:, 1])[:, None] - np.array(group_fars_mean[:, 1])[None, :], 3)
    frrs_1perc_diff = np.round(np.array(group_frrs_mean[:, 1])[:, None] - np.array(group_frrs_mean[:, 1])[None, :], 3)
    fars_diffTOT = np.round(np.array(group_fars_mean[:, 2])[:, None] - np.array(group_fars_mean[:, 2])[None, :], 3)
    frrs_diffTOT = np.round(np.array(group_frrs_mean[:, 2])[:, None] - np.array(group_frrs_mean[:, 2])[None, :], 3)
    fars_1perc_diffTOT = np.round(np.array(group_fars_mean[:, 3])[:, None] - np.array(group_fars_mean[:, 3])[None, :],
                                  3)
    frrs_1perc_diffTOT = np.round(np.array(group_frrs_mean[:, 3])[:, None] - np.array(group_frrs_mean[:, 3])[None, :],
                                  3)
    titles = ["FAR Diff. (Group AVG)", "FRR Diff. (Group AVG)",
              "FAR Diff. (Group AVG - far 1%)", "FRR Diff. (Group AVG - far 1%)",
              "FAR Diff. (Whole set AVG)", "FRR Diff. (Whole set AVG)",
              "FAR Diff. (Whole set AVG - far 1%)", "FRR Diff. (Whole set AVG - far 1%)"]
    data = [fars_diff, frrs_diff, fars_1perc_diff, frrs_1perc_diff, fars_diffTOT, frrs_diffTOT, fars_1perc_diffTOT,
            frrs_1perc_diffTOT]
    d = 0
    for r in range(rows):
        for c in range(cols):
            axs[r, c].set_title(titles[d])
            m = axs[r, c].matshow(data[d])
            axs[r, c].set_box_aspect(1.0)
            axs[r, c].set_xticks(list(range(len(labels))), labels)
            axs[r, c].set_yticks(list(range(len(labels))), labels)
            w, h = data[d].shape
            fig.colorbar(m, ax=axs[r, c])
            for i in range(w):
                for j in range(h):
                    t = data[d][i, j]
                    axs[r, c].text(i, j, str(t), va='center', ha='center')
            d += 1
    fig.savefig(os.path.join(path, 'heatmaps_v2.png'), dpi=fig.dpi)
    plt.close('all')


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
    for comparison in comparisons:
        comparison = comparison.rstrip("\n")
        model_name = args.model.split("/")[-2]
        directory = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        directory = directory + "-" + model_name + "-" + str(comparison.split("/")[-1])
        os.makedirs(directory)
        if comparison.endswith(".csv"):
            print("Preparing test files...")
            dl_ref, dl_cmp, group_ref, group_cmp, ref_id = csv_to_test(comparison, bp)
            print("Inferencing dataset...")
            ref, cmp, matches = inference(dl_ref, dl_cmp, device, model)
            m = matches.flatten()
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            results = cos(ref, cmp)
            np_res = np.vstack([results.numpy(), m.numpy(), group_ref, group_cmp, list(map(int, ref_id))]).T
            outcos = os.path.join(directory, "inferenced_cosines", "_".join([model_name, "_results_", comparison, ".npy"]))
            np.save(outcos, np_res)
        elif comparison.endswith(".npy"):
            data = torch.tensor(np.load(comparison))
            results = data[:, 0]
            m = data[:, 1]
            group_ref = data[:, 2].numpy().astype(np.int32).tolist()
            ref_id = data[:, 4].numpy().astype(np.int32).tolist()
        else:
            raise Exception
        print("Computing and saving results...")
        eer_auc, groups = get_graphs_and_matrices(results, group_ref, m, directory)
        get_heatmaps(eer_auc, directory, groups)
        get_heatmaps_by_id_and_group(results, group_ref, ref_id, m, directory)


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
        parser.add_argument('--base', metavar='path', required=True,
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
