import cv2
import numpy as np
import torchvision.transforms as T
import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddSaltAndPepper(object):
    def __init__(self, proba):
        self.proba = proba

    def __call__(self, tensor):
        black = torch.tensor([0, 0, 0], dtype=torch.uint8)
        white = torch.tensor([255, 255, 255], dtype=torch.uint8)
        row, col, _ = tensor.shape
        probs = torch.rand((row, col))
        tensor[probs < (self.proba / 2)] = black
        tensor[probs > 1 - (self.proba / 2)] = white
        return tensor


def apply_noise(img, params):
    noise_type = params["type"]
    if noise_type == "gaussian_blur":
        sigma = params['sigma']
        f_size = int((2 * np.ceil(2 * sigma)) + 1)
        gb = cv2.GaussianBlur(img, (f_size, f_size), sigma, sigma)
        return gb
    if noise_type == "brightness":
        beta = params['beta']
        img += beta
        clip_img = np.clip(img, 0, 255)
        return clip_img
    if noise_type == "gaussian_noise":
        sigma = params['sigma']
        row, col, _ = img.shape
        mean = 0
        gaussian = np.random.normal(mean, sigma, (row, col))
        noisy_image = img + gaussian
        return noisy_image
    if noise_type == "S&P":
        p = params['probability']
        output = img.copy()
        black = np.array([0, 0, 0], dtype='uint8')
        white = np.array([255, 255, 255], dtype='uint8')
        row, col, _ = img.shape
        probs = np.random.random((row, col))
        output[probs < (p / 2)] = black
        output[probs > 1 - (p / 2)] = white
        return output


def get_noise_torch(params):
    if params is None:
        return None
    noise_type = params["type"]
    if noise_type == "gaussian_blur":
        sigma = params['sigma']
        f_size = int((2 * torch.ceil(2 * torch.tensor(sigma))) + 1)
        transform = T.Compose([
            T.ToPILImage(),
            T.GaussianBlur(kernel_size=(f_size, f_size), sigma=(sigma, sigma))
        ])
        return transform
    if noise_type == "brightness":
        beta = params['beta']
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: torch.clip(x.add(beta), min=0, max=255)),
            T.ToPILImage()
        ])
        return transform
    if noise_type == "gaussian_noise":
        sigma = params['sigma']
        transform = T.Compose([
            T.ToTensor(),
            AddGaussianNoise(mean=0., std=float(sigma)),
            T.ToPILImage()
        ])
        return transform
    if noise_type == "S&P":
        p = params['probability']
        transform = T.Compose([
            AddSaltAndPepper(p),
            T.ToPILImage()
        ])
        return transform


