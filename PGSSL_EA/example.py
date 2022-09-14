# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import datetime
import logging
import time
time_start = time.time()
import logging
from yacs.config import CfgNode as CN
from train_util import *
from DataRead_Dir import VOCDataset,creat_mask
from torch.utils.data import DataLoader
from network import BYOL
import sys

import segmentation_models_pytorch as smp

import torch.nn as nn
import sys

# from utils.calculate_weights import calculate_weigths_labels
import numpy as np

# from DataRead import MyDataset
from torchvision.transforms import transforms


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by Z_Segment_smp_Net_Google_WuHan _TRAIN for Z_Segment_smp_Net_Google_WuHan training parameter,
# or _TEST for Z_Segment_smp_Net_Google_WuHan test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------


class config():
    _C = CN()

    _C.MODEL = CN()
    _C.MODEL.DEVICE = "cuda"

    _C.MODEL.WEIGHT = ""
    _C.MODEL.PRETRAINED_MODELS = "pretrained_models"

    # ---------------------------------------------------------------------------- #
    # Solver
    # ---------------------------------------------------------------------------- #
    _C.SOLVER = CN()
    _C.SOLVER.MAX_ITER = 80000

    _C.SOLVER.BASE_LR = 0.003
    _C.SOLVER.BIAS_LR_FACTOR = 2

    _C.SOLVER.MOMENTUM = 0.9

    _C.SOLVER.WEIGHT_DECAY = 0.0005
    _C.SOLVER.WEIGHT_DECAY_BIAS = 0

    _C.SOLVER.GAMMA = 0.5
    _C.SOLVER.STEPS = (2000, 15000, 80000)

    _C.SOLVER.WARMUP_FACTOR = 1.0 / 3
    _C.SOLVER.WARMUP_ITERS = 2500
    _C.SOLVER.WARMUP_METHOD = "linear"

    _C.SOLVER.CHECKPOINT_PERIOD = 500

    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
    # see 2 images per batch
    _C.SOLVER.IMS_PER_BATCH = 1

    # ---------------------------------------------------------------------------- #
    # Specific test options
    # ---------------------------------------------------------------------------- #
    _C.TEST = CN()
    _C.TEST.EXPECTED_RESULTS = []
    _C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 8
    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
    # see 2 images per batch
    _C.TEST.IMS_PER_BATCH = 4

    # ---------------------------------------------------------------------------- #

    _C.DATALOADER = CN()
    _C.DATALOADER.NUM_WORKERS = 1
    _C.DATALOADER.SIZE_DIVISIBILITY = 0

    # Misc options
    # ---------------------------------------------------------------------------- #
    _C.OUTPUT_DIR = "models/"

    _C.DataSet = CN()
    _C.DataSet.TRAIN_NAME = ''
    _C.DataSet.TEST_NAME = ''
    _C.DataSet.ROOT_DIR = ""
    _C.DataSet.EXP_NAME = 'UNet'

    _C.DataSet.DATA_NAME = 'ChangFang'
    _C.DataSet.DATA_AUG = False
    _C.DataSet.DATA_WORKERS = 1
    _C.DataSet.DATA_RESCALE = 512

    _C.DataSet.DATA_RANDOMCROP = 500
    _C.DataSet.DATA_RANDOMROTATION = 45
    _C.DataSet.DATA_RANDOMSCALE = 1
    _C.DataSet.DATA_RANDOM_H = 20
    _C.DataSet.DATA_RANDOM_S = 20
    _C.DataSet.DATA_RANDOM_V = 20
    _C.DataSet.DATA_RANDOMFLIP = 0.5

    _C.DataSet.MODEL_SAVE_DIR = os.path.join('Model', _C.DataSet.EXP_NAME)

    _C.DataSet.TRAIN_BATCHES = 4
    _C.DataSet.TRAIN_SHUFFLE = True
    _C.DataSet.MODEL_NUM_CLASSES = 2

    _C.DataSet.TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    _C.DataSet.TEST_FLIP = True

    _C.DataSet.TEST_GPUS = 1
    _C.DataSet.TEST_BATCHES = 4






import glob
import matplotlib.pyplot as plt
if __name__ == '__main__':

    cfg = config()._C
    cfg.DataSet.TRAIN_NAME = '../Data/Satellite dataset Ⅱ (East Asia)/all'
    device = torch.device('cuda:0')

    model = BYOL()
    modelname = 'model_train_byol_psepud_all/model_0058000_f1_0.809_r0.780_p0.842.pth'
    dict = torch.load(modelname)['model']
     # print(dict['unet.encoder.conv1.weight'][0,0,0,0:7])
    # print(dict['encoder_q.conv1.weight'][0,0,0,0:7])
    # print(dict['recoder.encoder.conv1.weight'][0,0,0,0:7])
    # print(dict['encoder_k.conv1.weight'][0,0,0,0:7])
    # for k,v in dict.items():
    #     print(k)
    # exit()


    model.load_state_dict(dict)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    val_dataset = VOCDataset('../Data/Satellite dataset Ⅱ (East Asia)/test', cfg.DataSet, 'val', cfg.DataSet.DATA_AUG)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    model.eval()


    for iteration, sample_batched in enumerate(val_loader, 0):
        images= sample_batched['image']

        images = images.to(device)


        mask = creat_mask(cfg.DataSet.DATA_RESCALE,4,0.4)
        b,c,_,_ = images.shape

        mask = torch.tensor(mask).to(device)
        mask_image = (mask * images).float()


        p1, p2, z1, z2,result1,result2,result3 = model(x1=images, x2=images,x3 = mask_image)
        result1 = torch.argmax(result1,dim=1)

        print(result3.shape)


        plt.subplot(2,2,1)
        plt.imshow(images[0].cpu().detach().numpy().transpose(1,2,0))
        plt.subplot(2,2,2)
        plt.imshow(mask_image[0].cpu().detach().numpy().transpose(1,2,0))
        plt.subplot(2,2,3)
        plt.imshow(result3[0].cpu().detach().numpy().transpose(1,2,0))
        plt.subplot(2,2,4)
        plt.imshow(mask.cpu().numpy())
        plt.show()

