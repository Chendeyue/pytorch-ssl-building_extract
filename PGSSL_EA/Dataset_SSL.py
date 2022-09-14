'''
@anthor: Wenyuan Li
@desc: Datasets for self-supervised
@date: 2020/5/15
'''

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import random
import matplotlib.pyplot as plt

import numpy as np
from torchvision import transforms
import torchvision
import math
from torchvision.transforms import Compose,RandomResizedCrop
from PIL import ImageFilter
# from transform_file import ContrastiveCrop
import torchvision.transforms.functional as F

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

from torch.distributions.beta import Beta
class ContrastiveCrop(RandomResizedCrop):  # adaptive beta
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        # a == b == 1.0 is uniform distribution
        self.beta = Beta(alpha, alpha)

    def get_params(self, img, box, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        # width, height = F._get_image_size(img)
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                h0, w0, h1, w1 = box
                ch0 = max(int(height * h0) - h//2, 0)
                ch1 = min(int(height * h1) - h//2, height - h)
                cw0 = max(int(width * w0) - w//2, 0)
                cw1 = min(int(width * w1) - w//2, width - w)

                i = ch0 + int((ch1 - ch0) * self.beta.sample())
                j = cw0 + int((cw1 - cw0) * self.beta.sample())
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, box):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, box, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as Z_Segment_smp_Net_Google_WuHan positive pair.
    """

    def __init__(self, size):
        s = 0.5
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                # ContrastiveCrop(alpha=1, size=size, scale=(0.2, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        # print(x)
        return self.train_transform(x), self.train_transform(x)



class VOCDataset_SSL(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        # self.root_dir = os.path.join(cfg.ROOT_DIR, 'data', 'VOCdevkit')

        self.dataset_dir = dataset_name

        self.img_dir = self.dataset_dir

        self.name_list = [name for name in os.listdir(self.img_dir)]
        self.transform = TransformsSimCLR(size=224)
        self.boxes = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file = self.img_dir + '/' + name

        image = Image.open(img_file)

        box = self.boxes[idx].float().tolist()

        x_i,x_j= self.transform(image)

        return x_i,x_j



if __name__ == '__main__':

    dirname = '../Data/谷歌数据集/train/images/'


    Dataset = VOCDataset_SSL(dirname)

    for x_i,x_j in Dataset:
        plt.subplot(1, 2, 1)
        plt.imshow(x_i.numpy().transpose(1,2,0))
        plt.subplot(1, 2, 2)
        plt.imshow(x_j.numpy().transpose(1,2,0))
        plt.show()