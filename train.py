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
import shutil
import tqdm
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
from utils.logger import TensorBoardLogger
from utils.metric import calculate_score


def train_epoch():

    model.train()
    logger.start_train_epoch()
    for images, targets, image_ids in tqdm.tqdm(train_data_loader):
        images = list(image.float().to(device) for image in images)

        # なぜか model(images, targets)を実行するとtargets内のbounding boxの値が変わるため値を事前に退避...
        target_boxes = [target['boxes'].detach().cpu().numpy().astype(np.int32) for target in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_dict_detach = {k: v.cpu().detach().numpy() for k, v in loss_dict.items()}
        logger.send_loss(loss_dict_detach)
    
    logger.send_images(images, image_ids, target_boxes, None)
    logger.end_train_epoch()


def evaluate_epoch():

    logger.start_valid_epoch()
    with torch.no_grad():
        for images, targets, image_ids in valid_data_loader:
            model.train()
            optimizer.zero_grad()
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # なぜか model(images, targets)を実行するとtargets内のbounding boxの値が変わるため値を事前に退避...
            target_boxes = [target['boxes'].detach().cpu().numpy().astype(np.int32) for target in targets]
            loss_dict = model(images, targets)
            loss_dict_detach = {k: v.cpu().detach().numpy() for k, v in loss_dict.items()}
            logger.send_loss(loss_dict_detach) 

            # Start calculating scores for competition
            model.eval()
            outputs = model(images)
            matric_score = calculate_score(outputs, targets)
            logger.send_score(matric_score)

    # 最後のevalのloopで生成されたものを保存する
    logger.send_images(images, image_ids, target_boxes, outputs)
    logger.end_valid_epoch()


def update_dict(d, keys, value):
        if len(keys) == 1:
            d[keys[0]] = value 
        else:
            update_dict(d[keys[0]], keys[1:], value)


def expand_json(d, keys=[]):
    for key in d.keys():
        if key == 'tuple':
            update_value = tuple([v for v in d[key].values()])
            update_dict(config, keys, update_value)
            continue
        elif key == 'list':
            update_value = list([v for v in d[key].values()])
            update_dict(config, keys, update_value)
            continue
        elif type(d[key]) is dict:
            expand_json(d[key], keys=keys+[key])
    return d


if __name__ == '__main__':

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('json_path')
    args = parser.parse_args()

    assert os.path.exists(args.json_path), "json\'s name '%s' is not valid." % args.json_path

    with open(args.json_path, 'r') as f:
        config = json.load(f)
    config = expand_json(config)

    SRC_DIR = Path('.').resolve()/'src'
    TRAIN_IMAGE_DIR = SRC_DIR/'train'
    TEST_IMAGE_DIR= SRC_DIR/'test'
    DATAFRAME = convert_dataframe( pd.read_csv(str(SRC_DIR/'train.csv')))
    OUTPUT_DIR = Path('.').resolve()/'output'/config['general']['output_dirname']

    device = torch.device('cuda:0')

    if os.path.exists(OUTPUT_DIR):
        print('[WIP]: reload weights. Execute sys.exit().')
        sys.exit()  # [WIP]: reload weights
    else:

        # copy json to output dir
        os.makedirs(str(OUTPUT_DIR), exist_ok=False)
        shutil.copy(args.json_path , str(OUTPUT_DIR/"config.json"))

        debug = config['debug']
        random_seed = config['general']['seed']
        model_save_interval = config['general']['model_save_interval']
        trained_epoch = 0
        trained_iter = 0
        trained_weights_path = None        
    
    # set seed (not enough for complete reproducibility)
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  
    torch.cuda.manual_seed(random_seed)  
    
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

    train_dataset = GWDDataset(train_dataframe, TRAIN_IMAGE_DIR, config, is_train=True)
    valid_dataset = GWDDataset(valid_dataframe, TRAIN_IMAGE_DIR, config, is_train=False)

    train_data_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=1, collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate_fn)

    # load model and make parallel
    model = get_model(config['model']).to(device)
    # model = torch.nn.DataParallel(model) 
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(config['train']['optimizer'], trainable_params)
    scheduler = get_scheduler(config['train']['scheduler'], optimizer)

    logger = TensorBoardLogger(model, optimizer, output_dir=OUTPUT_DIR, trained_epoch=trained_epoch, model_save_interval=model_save_interval)

    # training
    for epoch in range(trained_epoch+1, config['train']['epochs']+1):

        print("\r [Epoch %d]" % epoch)

        train_epoch()
        evaluate_epoch()
        if scheduler is not None:
            scheduler.step()


