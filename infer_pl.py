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

######################################################################
# This inference code is to help you visualize the CAMs results
# To generate pGT for SS (second phase), refer https://github.com/jbeomlee93/RIB.
######################################################################

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    
    # Augmentation
    parser.add_argument("--resize", default=[384,512], nargs='+', type=float)
    parser.add_argument("--crop", default=[256,256], nargs='+', type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)


    # Hyper-parameters
    parser.add_argument("--D", default=256, type=int)
    parser.add_argument("--grid", default=16, type=int)
    parser.add_argument("--prob", default=20, type=int)
    parser.add_argument("--W", default=[1, 0.5, 0.7, 0.3], nargs='+', type=float)

    # Experiments
    parser.add_argument("--exp", default='recon_cvpr23', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--date", required=True, type=str)
    parser.add_argument("--phase", default='infer', type=str)
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--cam", action='store_true')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--alphas", default=[6,24], nargs='+', type=int)
    
    args = parser.parse_args()
    
    infer_dataset = utils.build_dataset_moco(args, phase='val', path=args.infer_list)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, pin_memory=True)
       
    model = getattr(importlib.import_module('models.exp_'+args.exp), 'Exp')(args)
    
    dict_dir = os.path.join('./experiments',args.date+"_"+args.name,'version_1','checkpoints','last.ckpt')
    model.load_pretrained(dict_dir)

    vis_path = None
    cam_path = None
    crf_path = None

    base_path = './results/' + args.name
    os.makedirs(base_path, exist_ok=True)
    if args.vis:
        vis_path = base_path + '/vis'
        os.makedirs(vis_path, exist_ok=True)
    if args.cam:
        cam_path = base_path + '/cam'
        os.makedirs(cam_path, exist_ok=True)
    if args.crf:
        crf_path = base_path + '/crf'
        os.makedirs(crf_path, exist_ok=True)
        for a in args.alphas:
            os.makedirs(crf_path+'/'+str(a).zfill(2), exist_ok=True)

    for batch in tqdm(infer_data_loader):
        model.infer(batch, vis_path=vis_path, cam_path=cam_path, crf_path=crf_path, alphas=args.alphas)