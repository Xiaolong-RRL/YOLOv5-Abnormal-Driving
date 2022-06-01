import argparse
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.common import *
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import check_dataset
from utils.torch_utils import ModelEMA

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # 这个 Worker 是这台机器上的第几个 Worker
RANK = int(os.getenv('RANK', -1))  # 这个 Worker 是全局第几个 Worker 对于单机多卡来说 LOCAL_RANK= RANK
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # 总共有几个 Worker


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    # yolov5s-voc/yolov5s-ghostconv-bifpn1-ca-voc
    parser.add_argument('--cfg', type=str, default='../models/yolov5s-bifpn-new.yaml',
                        help='model.yaml path')
    parser.add_argument('--data', type=str, default='../data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='../data/hyps/hyp.scratch-high.yaml',
                        help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--save-dir', default='train_test', help='save to project/name')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(hyp, opt, device):
    cfg = opt.cfg
    data = opt.data
    imgsz = opt.imgsz
    epochs = opt.epochs
    batch_size = opt.batch_size
    save_dir = Path(opt.save_dir)
    # Directories 保存训练结果的weights文件
    w = save_dir / 'weights'  # weights dir
    # 递归删除之前的文件夹，并新建一个
    try:
        shutil.rmtree(w)
    except OSError:
        pass
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters 加载超参数
    if isinstance(hyp, str):
        with open(hyp, encoding='utf-8', errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict 字典形式

    # 检查数据集 如果本地没有 则从torch库中下载并解压数据集
    data_dict = check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']  # 获取训练数据集和验证集的路径
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names

    # 加载模型
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    params = list(model.named_parameters())  # get the index by debuging
    names = model.state_dict()
    # print(params[122][0])  # name
    # print(params[122][1].data)  # data
    # print(model.state_dict()['model.13.w1'])
    # 获取模型总步长 Detect层输出层数和模型输入图片分辨率
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)

    # Optimizer
    # nbs 标称的batch_size 模拟的batch_size 比如默认的话上面设置的opt.batch_size=16 -> nbs=64
    # 也就是模型梯度累计 64/16=4(accumulate) 次之后就更新一次模型 等于变相的扩大了batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    # 将模型参数分为三组(weights、biases、bn)来进行分组优化
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        # hasattr: 测试指定的对象是否具有给定的属性，返回一个布尔值
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
        # Add BiFPN
        elif isinstance(v, BiFPN_Add2) and hasattr(v, 'w') and isinstance(v.w, nn.Parameter):
            g1.append(v.w)
        elif isinstance(v, BiFPN_Add3) and hasattr(v, 'w') and isinstance(v.w, nn.Parameter):
            g1.append(v.w)

        # elif isinstance(v, BiFPN_Concat) and hasattr(v, 'w1') and isinstance(v.w1, nn.Parameter):
        #     g1.append(v.w1)
        # elif isinstance(v, BiFPN_Concat) and hasattr(v, 'w2') and isinstance(v.w2, nn.Parameter):
        #     g1.append(v.w2)

    # 选择优化器 默认为SGD 并设置三种参数的优化方式
    optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2

    # Scheduler YOLOv5-6.1默认使用linear学习率
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # 实例化 scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    compute_loss = ComputeLoss(model)  # init loss class
    # EMA 单卡训练: 使用EMA(指数移动平均)对模型的参数做平均
    # 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒(减少模型的抖动)
    ema = ModelEMA(model)

    # 设置训练相关参数
    start_epoch, best_fitness = 0, 0.0
    # Trainloader 加载训练数据集
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size, gs,
                                              hyp=hyp, augment=True, cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    main(opt.hyp, opt, device)
