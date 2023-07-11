from __future__ import division
from __future__ import print_function

# Base
import os
import os.path as osp
from tqdm import tqdm
import random
import importlib
import argparse
import logging
import pdb
import datetime

from matplotlib import pyplot as plt

# DL
import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom
import tools.imutils as imutils
import tools.utils as utils
import tools.pyutils as pyutils
from evaluation import eval_in_script

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    # Augmentation
    parser.add_argument("--resize", default=[384,512], nargs='+', type=float)
    parser.add_argument("--crop", default=[256,256], nargs='+', type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)

    # Hyper-parameters
    parser.add_argument("--D", default=256, type=int)
    parser.add_argument("--grid", default=16, type=int)
    parser.add_argument("--prob", default=20, type=int)
    parser.add_argument("--W", default=[1, 0.2, 0.8, 0.3], nargs='+', type=float)

    # Learning rate
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--max_epochs", default=40, type=int)

    # Experiments
    parser.add_argument("--exp", default='final', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--seed", default=4242, type=int)
    parser.add_argument("--phase", default='train', type=str)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(6101)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_dataset = utils.build_dataset_moco(args, phase='train', path=args.train_list)
    # train_dataset, _, _ = torch.utils.data.random_split(train_dataset, [16,2,len(train_dataset)-18]) # For debug only
    val_dataset = utils.build_dataset_moco(args, phase='val', path=args.val_list)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    val_data_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True)
    
    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)
    max_step = train_num_img // args.batch_size * args.max_epochs
    args.max_step = max_step

    dt = datetime.datetime.now()
    name_string = str(dt.month).zfill(2) + str(dt.day).zfill(2) + '_' + args.name
    logger = TensorBoardLogger('./experiments/', name=name_string, default_hp_metric=True)
    os.makedirs(logger.log_dir+'/dict', exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, save_last=True)

    trainer = pl.Trainer(max_epochs=100, gpus=1, logger=logger, callbacks=[checkpoint_callback])
    model = getattr(importlib.import_module('models.exp_'+args.exp), 'Exp')(args)
    trainer.fit(model, train_data_loader, val_data_loader)