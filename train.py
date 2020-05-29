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
import copy
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
from torch.utils.data import WeightedRandomSampler
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
from model import get_model, Model
from optimizer import get_optimizer
from scheduler import get_scheduler

from utils.dataset import GWDDataset, collate_fn
from utils.transform import Transform
from utils.logger import Logger
from utils.functions import convert_dataframe, format_config_by_baseconfig, randomname
from utils.metric import calculate_score_for_each
from utils.postprocess import postprocessing
from utils.sampler import get_sampler
from utils.train_valid_split import train_valid_split

warnings.simplefilter('ignore')  # 基本warningオフにしたい

def exec_train(config, train_data_loader, valid_data_loader, OUTPUT_DIR, fold, trained_epoch=0):
    # load model and make parallel
    device = torch.device('cuda:0')
    model = Model(config['model']).to(device)
    # model = get_model(config['model']).to(device)
    # model = torch.nn.DataParallel(model) 
    
    # train setting
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(config['train']['optimizer'], trainable_params)
    scheduler = get_scheduler(config['train']['scheduler'], optimizer)

    # log setting
    logger = Logger(model, optimizer, output_dir=OUTPUT_DIR, run_name=RUN_NAME, trained_epoch=trained_epoch, config=config, fold=fold+1)
    
    # training
    for epoch in range(trained_epoch+1, config['train']['epochs']+1):
        if config['general']['kfold'] < 0:
            print("\r [Epoch %d]" % epoch)
        else:
            print("\r [Fold %d : Epoch %d]" % (fold+1, epoch))

        train_epoch(model, train_data_loader, logger, optimizer)
        evaluate_epoch(model, valid_data_loader, logger, optimizer)
        if scheduler is not None:
            scheduler.step(logger.last_valid_loss)
    
    logger.finish_training()

def train_epoch(model, train_data_loader, logger, optimizer):

    model.train()
    logger.start_train_epoch()
    for images, targets, image_ids in tqdm.tqdm(train_data_loader):
        # なぜか model(images, targets)を実行するとtargets内のbounding boxの値が変わるため値を事前に退避...
        targets_copied = copy.deepcopy(targets)
        target_boxes = [target['boxes'].detach().cpu().numpy().astype(np.int32) for target in targets_copied]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        loss_dict_detach = {k: v.cpu().detach().numpy() for k, v in loss_dict.items()}
        logger.send_loss(loss_dict_detach)
    logger.send_images(images, image_ids, target_boxes, None)
    logger.end_train_epoch()


def evaluate_epoch(model, valid_data_loader, logger, optimizer):

    logger.start_valid_epoch()
    with torch.no_grad():
        for images, targets, image_ids in valid_data_loader:
            model.eval()
            optimizer.zero_grad()
            # なぜか model(images, targets)を実行するとtargets内のbounding boxの値が変わるため値を事前に退避...
            targets_copied = copy.deepcopy(targets)
            target_boxes = [target['boxes'].detach().cpu().numpy().astype(np.int32) for target in targets_copied]
            preds, loss_dict = model(images, targets)
            loss_dict_detach = {k: v.cpu().detach().numpy() for k, v in loss_dict.items()}
            logger.send_loss(loss_dict_detach) 
            original_metric_scores = calculate_score_for_each(preds, targets_copied)
            processed_outputs = postprocessing(copy.deepcopy(preds), config["valid"]) if 'valid' in config.keys() else outputs
            processed_metric_scores = calculate_score_for_each(processed_outputs, targets_copied)
                
            logger.send_score(original_metric_scores, 'original')
            logger.send_score(processed_metric_scores, 'processed')
    # 最後のevalのloopで生成されたものを保存する
    logger.send_images(images, image_ids, target_boxes, preds, processed_outputs)
    logger.end_valid_epoch()

def get_loader(train_dataframe, valid_dataframe, config):

    train_dataset = GWDDataset(train_dataframe, TRAIN_IMAGE_DIR, config, is_train=True)
    valid_dataset = GWDDataset(valid_dataframe, TRAIN_IMAGE_DIR, config, is_train=False)

    def worker_init_fn(worker_id):   
        random.seed(worker_id+random_seed)   
        np.random.seed(worker_id+random_seed)   
    
    train_data_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], sampler=get_sampler(train_dataset, config['train']), shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=collate_fn)    
    valid_data_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    return train_data_loader, valid_data_loader

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
    config = format_config_by_baseconfig(config)    
    config = expand_json(config)

    SRC_DIR = Path('.').resolve()/'src'
    TRAIN_IMAGE_DIR = SRC_DIR/'train'
    TEST_IMAGE_DIR= SRC_DIR/'test'
    DATAFRAME = convert_dataframe(pd.read_csv(str(SRC_DIR/'train.csv')))
    OUTPUT_DIR = Path('.').resolve()/'output'/config['general']['output_dirname']
    RUN_NAME = config['general']['experiment_name'] + '/' + randomname(5)

    print('Run Name: ', RUN_NAME)

    if OUTPUT_DIR.name == 'sample' and os.path.exists(OUTPUT_DIR):
        print("'output/sample' is be overwritten.")
        shutil.rmtree(OUTPUT_DIR)
    
    if os.path.exists(OUTPUT_DIR):        
        print('[WIP]: reload weights. Execute sys.exit().')
        sys.exit()  # [WIP]: reload weights
    else:

        # copy json to output dir
        os.makedirs(str(OUTPUT_DIR), exist_ok=False)
        if config['general']['kfold'] > 0:
            k = config['general']['kfold']
            for fold in range(k):
                os.makedirs(str(OUTPUT_DIR/'fold_{0}'.format(fold+1)), exist_ok=False)
        with open(str(OUTPUT_DIR/"config.json"), "w") as f:
            json.dump(config, f, indent=4)

        debug = config['debug']
        random_seed = config['general']['seed']
        trained_epoch = 0
        trained_iter = 0
        trained_weights_path = None        
    
    # set seed (not enough for complete reproducibility)
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  
    torch.cuda.manual_seed(random_seed) 
    
    train_and_valid_ids_list = train_valid_split(DATAFRAME, config)
    # execute training
    # iterate if k-fold
    for fold, (train_ids, valid_ids) in enumerate(train_and_valid_ids_list):
        train_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(train_ids), :]
        valid_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(valid_ids), :]
        train_data_loader, valid_data_loader = get_loader(train_dataframe, valid_dataframe, config)

        OUTPUT_DIR_FOLDED = OUTPUT_DIR if config['general']['kfold'] < 0 else OUTPUT_DIR/'fold_{0}'.format(fold+1)

        exec_train(copy.deepcopy(config), 
                   train_data_loader, 
                   valid_data_loader, 
                   OUTPUT_DIR_FOLDED, 
                   fold, 
                   trained_epoch=0)
