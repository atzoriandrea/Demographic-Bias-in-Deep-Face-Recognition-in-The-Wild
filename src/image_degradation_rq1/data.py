import os, sys
import numpy as np
import cv2
from noises import get_noise_torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as nnF
import torch

High_Data = ["/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/celea_60000_SFD",
             "/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/SRtrainset_2",
             "/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/vggface2/vggcrop_test_lp10",
             "/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/vggface2/vggcrop_train_lp10"]
Low_Data = ["/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/LOW/wider_lnew"]


def get_images_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                paths.append(os.path.join(root, file))
    return paths


class faces_data(Dataset):
    def __init__(self, data_hr, data_lr):
        self.hr_imgs = [os.path.join(d, i) for d in data_hr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_imgs = [os.path.join(d, i) for d in data_lr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_len = len(self.lr_imgs)
        self.lr_shuf = np.arange(self.lr_len)
        np.random.shuffle(self.lr_shuf)
        self.lr_idx = 0
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.imread(self.hr_imgs[index])
        lr = cv2.imread(self.lr_imgs[self.lr_shuf[self.lr_idx]])
        self.lr_idx += 1
        if self.lr_idx >= self.lr_len:
            self.lr_idx = 0
            np.random.shuffle(self.lr_shuf)
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["lr"] = self.preproc(lr)
        data["hr"] = self.preproc(hr)
        data["hr_down"] = nnF.avg_pool2d(data["hr"], 4, 4)
        return data
    
    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)


class high2low_data(Dataset):
    def __init__(self, data_hr):
        hr_temp = [get_images_full_path(path) for path in data_hr]
        hr = []
        [hr.extend(el) for el in hr_temp]
        self.hr_imgs = hr
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.resize(cv2.imread(self.hr_imgs[index]), (64, 64))
        data['path'] = self.hr_imgs[index]
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["hr"] = self.preproc(hr)
        return data

    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)


class discriminator_data(Dataset):
    def __init__(self, data_ori, data_gen=None, thirdparty=None, noise_func=None):
        ori_temp = [get_images_full_path(path) for path in data_ori]
        self.gen_imgs = None
        self.third_imgs = None
        self.apply_noise = get_noise_torch(noise_func)
        if data_gen is not None:
            gen_temp = [get_images_full_path(path) for path in data_gen]
            gen = []
            [gen.extend(el) for el in gen_temp]
            self.gen_imgs = gen
            self.gen_len = len(self.gen_imgs)
            self.gen_shuf = np.arange(self.gen_len)
            np.random.shuffle(self.gen_shuf)
            self.gen_idx = 0

        if thirdparty is not None:
            third_temp = [get_images_full_path(path) for path in thirdparty]
            third = []
            [third.extend(el) for el in third_temp]
            self.third_imgs = third
            self.third_len = len(self.third_imgs)
            self.third_shuf = np.arange(self.third_len)
            np.random.shuffle(self.third_shuf)
            self.third_idx = 0
        ori = []
        [ori.extend(el) for el in ori_temp]

        self.ori_imgs = ori

        self.ori_len = len(self.ori_imgs)
        self.ori_shuf = np.arange(self.ori_len)
        np.random.shuffle(self.ori_shuf)
        self.ori_idx = 0
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((16, 16)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_imgs)

    def __getitem__(self, index):
        data = {}
        ori = cv2.imread(self.ori_imgs[self.ori_shuf[self.ori_idx]])

        #ori = cv2.resize(ori, (16, 16))
        if self.gen_imgs is not None:
            gen = cv2.resize(cv2.imread(self.gen_imgs[self.gen_shuf[self.gen_idx]]), (16, 16))
            self.gen_idx += 1
            if self.gen_idx >= self.gen_len:
                self.gen_idx = 0
                np.random.shuffle(self.gen_shuf)
            data["generated"] = self.preproc(gen)

        if self.third_imgs is not None:
            third = cv2.resize(cv2.imread(self.third_imgs[self.third_shuf[self.third_idx]]), (16, 16))
            self.third_idx += 1
            if self.third_idx >= self.third_len:
                self.third_idx = 0
                np.random.shuffle(self.third_shuf)
            data["thirdparty"] = self.preproc(third)

        self.ori_idx += 1
        if self.ori_idx >= self.ori_len:
            self.ori_idx = 0
            np.random.shuffle(self.ori_shuf)
        if self.apply_noise is not None:
            ori = self.apply_noise(ori)
        data["original"] = self.preproc(ori)
        return data

"""
if __name__ == "__main__":
    data = faces_data(High_Data, Low_Data)
    loader = DataLoader(dataset=data, batch_size=16, shuffle=True)
    for i, batch in enumerate(loader):
        print("batch: ", i)
        lrs = batch["lr"].numpy()
        hrs = batch["hr"].numpy()
        downs = batch["hr_down"].numpy()

        for b in range(batch["z"].size(0)):
            lr = lrs[b]
            hr = hrs[b]
            down = downs[b]
            lr = lr.transpose(1, 2, 0)
            hr = hr.transpose(1, 2, 0)
            down = down.transpose(1, 2, 0)
            lr = (lr - lr.min()) / (lr.max() - lr.min())
            hr = (hr - hr.min()) / (hr.max() - hr.min())
            down = (down - down.min()) / (down.max() - down.min())
            cv2.imshow("lr-{}".format(b), lr)
            cv2.imshow("hr-{}".format(b), hr)
            cv2.imshow("down-{}".format(b), down)
            cv2.waitKey()
            cv2.destroyAllWindows()

    print("finished.")
"""