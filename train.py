import argparse
import os
from pathlib import Path
import random
import re
import sys
import gc
import six
import json
import time
import datetime
from logging import getLogger
from time import perf_counter
import warnings
import glob

import numpy as np 
from numpy.random.mtrand import RandomState
import pandas as pd 

from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import cv2

import matplotlib.pyplot as plt

# --- torch ---
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from skimage.transform import AffineTransform, warp
import sklearn.metrics

# --- albumentations ---
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

# --- my modules ---
from model import get_model
from optimizer import get_optimizer
from scheduler import get_scheduler

from utils.dataset import GWDDataset, collate_fn
from utils.transform import Transform
from utils.functions import convert_dataframe
from utils.logger import Averager


if __name__ == '__main__':

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser()

    # --- general ---
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', default=now)
    parser.add_argument('--seed', type=int, default=42)

    # --- model ---
    parser.add_argument('--model_name', type=str, default='fasterrcnn')  
    parser.add_argument('--model_backborn', type=str, default='') 
    parser.add_argument('--model_not_pretrained', action='store_false')  

    # --- train ---
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--schedular_name', type=str, default='')

    # --- argument ---
    parser.add_argument('--blur_p', type=float, default=1.0)
    parser.add_argument('--brightness_contrast_p', type=float, default=1.0)
    parser.add_argument('--test_time_augment', action='store_true')

    args = parser.parse_args()

    SRC_DIR = Path('.').resolve()/'src'
    TRAIN_IMAGE_DIR = SRC_DIR/'train'
    TEST_IMAGE_DIR= SRC_DIR/'test'
    DATAFRAME = convert_dataframe( pd.read_csv(str(SRC_DIR/'train.csv')))
    OUTPUT_DIR = Path('.').resolve()/'output'/args.output_dir

    device = torch.device('cuda:0')

    if os.path.exists(OUTPUT_DIR):
        sys.exit()  # [WIP]: reload weights
    else:
        debug = args.debug
        random_seed = args.seed
        model_name = args.model_name
        model_backborn = args.model_backborn
        if model_backborn == '':
            if model_name == 'fasterrcnn':
                model_backborn = 'fasterrcnn_resnet50_fpn'
        model_pretrained = args.model_not_pretrained
        batch_size = args.batch_size
        epochs = args.epochs
        optimizer_name = args.optimizer_name
        initial_lr = args.lr
        b1 = args.b1
        b2 = args.b2
        schedular_name = args.schedular_name
        blur_p = args.blur_p
        brightness_contrast_p = args.brightness_contrast_p
        test_time_augment = args.test_time_augment
 
        trained_epoch = 0
        trained_iter = 0
        trained_weights_path = None        
    
    train_config = {
        'debug': debug,
        'output_dir': str(OUTPUT_DIR),
        'random_seed': random_seed,
        'model_name': model_name,
        'model_backborn': model_backborn,
        'model_pretrained': model_pretrained,
        'batch_size': batch_size,
        'epochs': epochs,
        'optimizer_name': optimizer_name,
        'initial_lr': initial_lr,
        'b1': b1, 
        'b2': b2,
        'schedular_name': schedular_name,
        'trained_epoch': trained_epoch,
        'trained_iter': trained_iter,
        'trained_weights_path': trained_weights_path if trained_weights_path else "",
        'test_time_augment': test_time_augment,
        'datetime': now
    }

    train_augment_config = {
        'blur_p': blur_p,
        'brightness_contrast_p': brightness_contrast_p,
    }

    if test_time_augment:
        valid_augment_config = train_augment_config
    else:
        valid_augment_config = {
            'blur_p': 0.,
            'brightness_contrast_p': 0.,
        }

    # set seed (not enough for complete reproducibility)
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  
    torch.cuda.manual_seed(random_seed)  
    
    # config for logging
    config = train_config
    for key, value in train_augment_config.items():
        config['train-' + key] = value
    for key, value in valid_augment_config.items():
        config['valid-' + key] = value

    
    # prepare for training
    image_ids = DATAFRAME['image_id'].unique()
    image_num = len(image_ids)
    image_ids = np.random.permutation(image_ids)

    train_data_size = int(image_num * 0.8) if not(debug) else 100
    valid_data_size = int(image_num - train_data_size) if not(debug) else 20

    train_ids = image_ids[:train_data_size]
    valid_ids = image_ids[train_data_size:(train_data_size+valid_data_size)]

    train_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(train_ids), :]
    valid_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(valid_ids), :]

    train_dataset = GWDDataset(train_dataframe, TRAIN_IMAGE_DIR, Transform(train_augment_config))
    valid_dataset = GWDDataset(valid_dataframe, TRAIN_IMAGE_DIR, Transform(valid_augment_config))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # load model and make parallel
    model = get_model(config)  
    model = torch.nn.DataParallel(model) 
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(config, trainable_params)
    scheduler = get_scheduler(config, optimizer)

    # training
    for epoch in range(trained_epoch+1, epochs+1):

        # train phase
        model.train()
        for images, targets, image_ids in train_data_loader:
            # import pdb; pdb.set_trace()
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # import pdb; pdb.set_trace()
            loss_dict = model(images, targets)



            import pdb; pdb.set_trace()