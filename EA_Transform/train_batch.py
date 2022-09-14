# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import datetime
import logging
import time
time_start = time.time()
import logging
from yacs.config import CfgNode as CN
from train_util import *
from DataRead_Dir import VOCDataset
from torch.utils.data import DataLoader
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
    _C.SOLVER.MAX_ITER = 80003

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
    _C.SOLVER.IMS_PER_BATCH = 4

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


def do_train(
        model,
        logger,
        data_loader,
        valloader,
        optimizer,
        scheduler,
        criterion,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):
    # logger = logging.getLogger("Unet.trainer")
    logger.info("Start training")

    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    logger.info("maxiter:{},satrt:{}".format(max_iter, start_iter))

    model.train()
    start_training_time = time.time()
    end = time.time()

    trainloss_sum = 0.0
    maxprecise = 0.0

    for iteration, sample_batched in enumerate(data_loader, start_iter):
        images, targets = sample_batched['image'], sample_batched['segmentation']
        data_time = time.time() - end

        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = targets.long().to(device)

        # print(images.shape,targets.shape,images.max(),targets.max())
        # print(images[0,0,0])
        # print(targets[0,0,0])
        # exit()
        losses = criterion(model(images), targets)
        trainloss_sum += losses
        meters.update(loss=losses)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "loss_sum:{loss}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f} M",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    loss=trainloss_sum,
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            trainloss_sum = 0.0

        if iteration % checkpoint_period == 0:
            precise = checkpointer.evalmodel(valloader)

            logger.info("precise:{}".format(precise))
            if precise[-1] > maxprecise:
                maxprecise = precise[-1]

                checkpointer.save("model_{:07d}_f1_{:.03f}_r{:.03f}_p{:.3f}".format(iteration, precise[-2],
                                                                                    precise[-4], precise[-3]),
                                  **arguments)

    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    precise = checkpointer.evalmodel(valloader)
    return [['test precise',precise[1],precise[2],precise[3]]]

def write_result(filename,precise,headname):
    with open(filename,"a") as f:
        f.writelines(headname+':\n')
        for item in precise:
            f.writelines(item[0]+":recall:[{:.3f}],recall:[{:.3f}],f1:[{:.3f}]\n".format(
                                                                          item[1],
                                                                          item[2],
                                                                          item[3]))  # 自带文件关闭功能，不需要再写f.close()
        f.writelines('\n\n')

def train(cfg,logger ):
    model = smp.Unet(classes=2)

    modelname = 'new/1.pth'
    dict = torch.load(modelname)['model']

    new_state_dict = OrderedDict()
    for k, v in dict.items():
        if 'unet.encoder.' in k:
            new_k = k.split('encoder.')[-1]
            # print(k, new_k)
            new_state_dict[new_k] = v
    # exit()
    model.encoder.load_state_dict(new_state_dict)

    # for parameter in model.encoder.parameters():
    #     parameter.requires_grad = False

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)



    logger.info("Running with config:\n{}".format(cfg))

    optimizer = make_optimizer(cfg, model)
    # 'allmodel/model_0050000_J.pth',
    scheduler = make_lr_scheduler(cfg, optimizer)


    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = True
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, logger
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    # logger.info("Load Succes:" + cfg.MODEL.WEIGHT)

    dataset = VOCDataset(cfg.DataSet.TRAIN_NAME, cfg.DataSet, 'train')

    data_loader = make_data_loader(
        cfg,
        dataset,
        is_train=True,
        is_distributed=False,
        start_iter=arguments["iteration"],
    )

    val_dataset = VOCDataset(cfg.DataSet.TEST_NAME, cfg.DataSet, 'val')

    valloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    weight = None

    logger.info("dataset weight:{} ".format(weight))

    criterion = nn.CrossEntropyLoss(weight=weight)
    logger.info("dataset long:{} , Val:{}".format(len(dataset), len(val_dataset)))

    precise = do_train(
        model,
        logger,
        data_loader,
        valloader,
        optimizer,
        scheduler,
        criterion,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return precise


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



def val_Data(modelname,device,valname,cfg):
    model = smp.Unet(classes=2)
    dict = torch.load(modelname)
    model.load_state_dict(dict['model'])
    device = torch.device(device)
    model.to(device)
    model.eval()

    correct = 0.0
    total = 0.0

    TT = 1e-8
    PT = 1e-8

    correct_t = 0.0

    val_dataset = VOCDataset(valname, cfg.DataSet, 'val')

    val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

    for i, sample_batched in enumerate(val_loader):
        images = sample_batched['image'].type(torch.FloatTensor).to(device)
        labels = sample_batched['segmentation'].type(torch.LongTensor).to(device)

        output = model(images)

        prediction = torch.argmax(output, dim=1)

        correct += (prediction == labels.long()).sum().float()
        total += labels.nelement()

        TT += labels.sum().float()
        PT += prediction.sum().float()
        correct_t += (prediction * labels).sum().float()

    ap = (correct / total).item()
    tt = (correct_t / TT).item() + 1e-6
    pt = (correct_t / PT).item() + 1e-6
    f1 = 2 * pt * tt / (pt + tt + 1e-6)

    iou = (correct_t / (TT + PT - correct_t)).item()

    return [["Val precise:",tt,pt,f1]]


def train_dir(outdir,trainname,device,name):
    cfg = config()._C

    cfg.OUTPUT_DIR = outdir
    Dir = cfg.OUTPUT_DIR
    if not os.path.exists(Dir):
        os.makedirs(Dir)

    cfg.DataSet.TRAIN_NAME = trainname
    cfg.DataSet.TEST_NAME = '../Data/Satellite dataset Ⅱ (East Asia)/val'
    cfg.MODEL.DEVICE = device
    # cfg.DataSet.TRAIN_NAME = '../Data/YJS_New/train_512/14400 (all)'
    logger = setup_logger(name, Dir, False)
    precise = train(cfg,logger)

    maxiter = cfg.SOLVER.MAX_ITER
    modelname = cfg.OUTPUT_DIR + "/model_{:07d}.pth".format(maxiter)

    result = val_Data(modelname, cfg.MODEL.DEVICE,'../Data/Satellite dataset Ⅱ (East Asia)/test',cfg)
    precise += result
    logger.info("Running Eval Val Over!!!")
    filename = 'result_new.txt'
    print(precise)
    write_result(filename,precise,name)
    # exit()
    # print(precise)

import glob
if __name__ == '__main__':
    # outdir = 'models_rain_0.01/'
    # trainname = '../Data/谷歌数据集/train_0.01'

    trainname_dir = glob.glob('../Data/Satellite dataset Ⅱ (East Asia)/train#0.01')
    # trainname_dir = ['../../Data/Satellite dataset Ⅱ (East Asia)/train',
    #                  '../../Data/Satellite dataset Ⅱ (East Asia)/train#0.01']
    out_dirs = ['model1_'+item.split('/')[-1] for  item in trainname_dir]

    for trainname,outdir in zip(trainname_dir,out_dirs):
        device = 'cuda:1'
        # if os.path.exists(outdir) :continue
        train_dir(outdir,trainname,device,outdir)
    print('cost time:',time.time()-time_start)