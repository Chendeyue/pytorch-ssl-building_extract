# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------



import multiprocessing
from skimage import io
from PIL import Image

from torch.utils.data import Dataset, DataLoader

'''
'--------------------------------------======VOCDataset =======----------------------
'''

import cv2
import numpy as np
import torch
import os
import time
time_start = time.time()

class Rescale(object):
    """Rescale the image in Z_Segment_smp_Net_Google_WuHan sample to Z_Segment_smp_Net_Google_WuHan given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, is_continuous=False, fix=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.fix = fix

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h, w):
            return sample

        if self.fix:
            h_rate = self.output_size[0] / h
            w_rate = self.output_size[1] / w
            min_rate = h_rate if h_rate < w_rate else w_rate
            new_h = h * min_rate
            new_w = w * min_rate
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)

        top = (self.output_size[0] - new_h) // 2
        bottom = self.output_size[0] - new_h - top
        left = (self.output_size[1] - new_w) // 2
        right = self.output_size[1] - new_w - left
        if self.fix:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation']
            seg = cv2.resize(segmentation, dsize=(new_w, new_h), interpolation=self.seg_interpolation)
            if self.fix:
                seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation'] = seg
        sample['image'] = img
        return sample


class Centerlize(object):
    def __init__(self, output_size, is_continuous=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h, w):
            return sample

        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        top = (new_h - h) // 2
        bottom = new_h - h - top
        left = (new_w - w) // 2
        right = new_w - w - left
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation']
            seg = cv2.copyMakeBorder(segmentation, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation'] = seg
        sample['image'] = img

        return sample


class RandomCrop(object):
    """Crop randomly the image in Z_Segment_smp_Net_Google_WuHan sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                left: left + new_w]

        segmentation = segmentation[top: top + new_h,
                       left: left + new_w]
        sample['image'] = image
        sample['segmentation'] = segmentation
        return sample


class RandomHSV(object):
    """Generate randomly the image in hsv space."""

    def __init__(self, h_r, s_r, v_r):
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def __call__(self, sample):
        image = sample['image']
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.int32)
        s = hsv[:, :, 1].astype(np.int32)
        v = hsv[:, :, 2].astype(np.int32)
        delta_h = np.random.randint(-self.h_r, self.h_r)
        delta_s = np.random.randint(-self.s_r, self.s_r)
        delta_v = np.random.randint(-self.v_r, self.v_r)
        h = (h + delta_h) % 180
        s = s + delta_s
        s[s > 255] = 255
        s[s < 0] = 0
        v = v + delta_v
        v[v > 255] = 255
        v[v < 0] = 0
        hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
        sample['image'] = image
        return sample


class RandomFlip(object):
    """Randomly flip image"""

    def __init__(self, threshold):
        self.flip_t = threshold

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        if np.random.rand() < self.flip_t:
            image_flip = np.flip(image, axis=1)
            segmentation_flip = np.flip(segmentation, axis=1)
            sample['image'] = image_flip
            sample['segmentation'] = segmentation_flip
        return sample


class RandomRotation(object):
    """Randomly rotate image"""

    def __init__(self, angle_r, is_continuous=False):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        m = cv2.getRotationMatrix2D(center=(col / 2, row / 2), angle=rand_angle, scale=1)
        new_image = cv2.warpAffine(image, m, (col, row), flags=cv2.INTER_CUBIC, borderValue=0)
        new_segmentation = cv2.warpAffine(segmentation, m, (col, row), flags=self.seg_interpolation, borderValue=0)
        sample['image'] = new_image
        sample['segmentation'] = new_segmentation
        return sample


class RandomScale(object):
    """Randomly scale image"""

    def __init__(self, scale_r, is_continuous=False):
        self.scale_r = scale_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_scale = np.random.rand() * (self.scale_r - 1 / self.scale_r) + 1 / self.scale_r
        img = cv2.resize(image, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(segmentation, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['segmentation'] = seg
        return sample





class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' in key:
                image = sample[key]
                image = image.transpose((2, 0, 1))
                sample[key] = torch.from_numpy(image.astype(np.float32) / 255.0)

            if 'origin' in key:
                image = sample[key]
                image = image.transpose((2, 0, 1))
                sample[key] = torch.from_numpy(image.astype(np.float32) / 255.0)


        return sample


def onehot(label, num):
    m = label
    one_hot = np.eye(num)[m]
    return one_hot


'''
--------------------------------------======VOCDataset =======----------------------
'''
import math

def creat_mask(imagesize = 384,finalsize = 12,mask_ratio = 0.2):

    if imagesize % finalsize !=0:
        print('mask 大小不符合要求')
        exit()

    poolsize = imagesize//finalsize

    ratio = mask_ratio
    length = poolsize * poolsize
    index = int(length * ratio)

    index_list = np.arange(length)
    np.random.shuffle(index_list)

    index_list = index_list[0:index]

    mask = np.ones((imagesize,imagesize))
    for idx in index_list:
        x = idx  % poolsize
        y = idx // poolsize
        mask[y*finalsize:(y+1)*finalsize,x*finalsize:(x+1)*finalsize] = 0

    return mask



class VOCDataset(Dataset):
    def __init__(self, dataset_name, cfg, period, aug):
        self.dataset_name = dataset_name

        # self.root_dir = os.path.join(cfg.ROOT_DIR, 'data', 'VOCdevkit')

        self.dataset_dir = dataset_name

        self.rst_dir = 'Photo/Segmentation/ChangF'
        # self.eval_dir = os.path.join(self.root_dir, 'eval_result', dataset_name, 'Segmentation')

        self.period = period
        self.img_dir = os.path.join(self.dataset_dir, 'image')
        self.seg_dir = os.path.join(self.dataset_dir, 'label')

        self.name_list = [name for name in os.listdir(self.img_dir)]

        self.rescale = None
        self.centerlize = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.multiscale = None
        self.totensor = ToTensor()
        self.cfg = cfg

        self.categories = [
            'Building',  # 1
            # 'bicycle',  # 2
        ]

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE, fix=False)
            # self.centerlize = Centerlize(cfg.DATA_RESCALE)
        if 'train' in self.period:
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file = self.img_dir + '/' + name
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        sample = {'image': image,'origin':image.copy()}

        seg_file = self.seg_dir + '/' + name
        segmentation = np.array(Image.open(seg_file))
        sample['segmentation'] = (segmentation > 0).astype(int)

        if 'train' in self.period:
            if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)

        sample = self.totensor(sample)

        # mask = creat_mask(self.cfg.DATA_RESCALE,8,0.4)
        # mask =  np.tile(mask[np.newaxis,:,:], (3,1,1))
        # sample['image'] = sample['image'] * mask

        return sample



import matplotlib.pyplot as plt
from train_batch import *

if __name__ == '__main__':
    arguments = {}
    arguments["iteration"] = 0

    cfg = config()._C

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = True


    dataset = VOCDataset('../Data/WHU/train', cfg.DataSet, 'train', cfg.DataSet.DATA_AUG)

    # dataloader = DataLoader(dataset,
    #                         batch_size=cfg.DataSet.TRAIN_BATCHES,
    #                         shuffle=cfg.DataSet.TRAIN_SHUFFLE,
    #                         num_workers=cfg.DataSet.DATA_WORKERS,
    #                         drop_last=True)
    print("time,", time.time() - time_start)

    data_loader = make_data_loader(
        cfg,
        dataset,
        is_train=True,
        is_distributed=False,
        start_iter=arguments["iteration"],
    )

    print("time,",time.time()-time_start)

    from train_util import InpaintingLoss
    fun = InpaintingLoss()

    for iteration, sample_batched in enumerate(data_loader, 0):
        images, targets = sample_batched['image'], sample_batched['origin']
        mask = creat_mask(512,8,0.4)
        mask = torch.tensor(mask)
        loss = fun(images,images,mask)
        # print(loss)
        # exit()
        # # print(images[:,:,mask].shape)
        # # exit()
        # images = images.permute(2,3,0,1)
        # print(images[mask].shape)
        # exit()
        # mask =  np.tile(mask[:,:,np.newaxis], (1, 1,3))
        # print(mask.shape)
        data = images * mask
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(images[0].numpy().transpose(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(data[0].numpy().transpose(1, 2, 0))
        plt.show()
