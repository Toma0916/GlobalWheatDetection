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
from collections import defaultdict


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

def sanity_check(loaded_models):
    random_seed_list = []
    train_valid_split_list = []

    if len(loaded_models.keys()) == 1:
        return True

    for key in loaded_models.keys():
        random_seed_list.append(loaded_models[key]['config']['general']['seed'])
        train_valid_split_list.append(loaded_models[key]['config']['general']['train_valid_split'])
    
    if 1 < len(set(random_seed_list)):
        return False
    
    for i in range(len(loaded_models.keys()) - 1):
        if train_valid_split_list[i] != train_valid_split_list[i+1]:
            return False

    return True
    

if __name__ == '__main__':

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    SRC_DIR = Path('.').resolve()/'src'
    TRAIN_IMAGE_DIR = SRC_DIR/'train'
    TEST_IMAGE_DIR= SRC_DIR/'test'
    DATAFRAME = convert_dataframe(pd.read_csv(str(SRC_DIR/'train.csv')))    
    OUTPUT_DIR = Path('.').resolve()/'output'


    # [WARN]: hard cording now
    pretrained_paths = ['manhattan/exp_9',
                        'manhattan/exp_13']

    loaded_models = defaultdict(dict)

    for path in pretrained_paths:
        config_path = OUTPUT_DIR/path/'config.json'
        with open(str(config_path), 'r') as f:
            config = json.load(f)

        if 'kfold' in config['general'] and 0 < config['general']['kfold']:
            print('[WARN]: not supporting k-fold. Execute sys.exit().')
            sys.exit()  # [WIP]: reload weights
        
        weight_path = list(sorted((OUTPUT_DIR/path).glob('*.pt')))[-1]  # latest weight

        loaded_models[path]['config'] = config
        loaded_models[path]['weight_path'] = weight_path
    
    # check random_seed and train_valid_split
    assert sanity_check(loaded_models), 'The models you selected are invalid.'
    
    
    random_seed = loaded_models[pretrained_paths[0]]['config']['general']['seed']

    # set seed (not enough for complete reproducibility)
    debug = True
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  
    torch.cuda.manual_seed(random_seed) 
    
    device = torch.device('cuda:0')

    def worker_init_fn(worker_id):   
        random.seed(worker_id+random_seed)   
        np.random.seed(worker_id+random_seed) 

    train_ids, valid_ids = train_valid_split(DATAFRAME, loaded_models[pretrained_paths[0]]['config'])[0]
    train_ids = train_ids[:100] if debug else train_ids
    valid_ids = valid_ids[:20] if debug else valid_ids
    train_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(train_ids), :]
    valid_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(valid_ids), :]
    train_dataset = GWDDataset(train_dataframe, TRAIN_IMAGE_DIR, loaded_models[pretrained_paths[0]]['config'], is_train=True)
    valid_dataset = GWDDataset(valid_dataframe, TRAIN_IMAGE_DIR, loaded_models[pretrained_paths[0]]['config'], is_train=False)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=collate_fn)    
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=collate_fn)

    # load model and predict
    train_predicts = defaultdict(lambda: {'boxes': np.array([]), 'target': np.array([]), 'scores': np.array([])})
    valid_predicts = defaultdict(lambda: {'boxes': np.array([]), 'target': np.array([]), 'scores': np.array([])})

    # predict train and valid
    for key in loaded_models.keys():
        model = Model(loaded_models[key]['config']['model'])
        model = model.load_state_dict(str(loaded_models[key]['weight_path'])).to(device)
        model.eval()
        loaded_models[path]['model'] = model

        for images, targets, image_ids in tqdm.tqdm(train_data_loader):
            image_id = image_ids[0]
            preds, loss_dict = model(images, targets)
            if train_predicts[image_id]['boxes'].shape[0] == 0:
                train_predicts[image_id]['boxes'] = preds[0]['boxes'].detach().cpu().numpy()
                train_predicts[image_id]['target'] = targets[0]['boxes'].detach().cpu().numpy()
                train_predicts[image_id]['scores'] = preds[0]['scores'].detach().cpu().numpy()
            else:
                train_predicts[image_id]['boxes'] = np.concatenate([train_predicts[image_id]['boxes'], preds[0]['boxes'].detach().cpu().numpy()], axis=0)
                train_predicts[image_id]['scores'] = np.concatenate([train_predicts[image_id]['scores'], preds[0]['scores'].detach().cpu().numpy()], axis=0)
                sorted_idx = np.argsort(train_predicts[image_id]['scores'])[::-1]
                train_predicts[image_id]['boxes'] = train_predicts[image_id]['boxes'][sorted_idx, :]
                train_predicts[image_id]['scores'] = train_predicts[image_id]['scores'][sorted_idx]
        
        for images, targets, image_ids in tqdm.tqdm(valid_data_loader):
            image_id = image_ids[0]
            preds, loss_dict = model(images, targets)
            if valid_predicts[image_id]['boxes'].shape[0] == 0:
                valid_predicts[image_id]['boxes'] = preds[0]['boxes'].detach().cpu().numpy()
                valid_predicts[image_id]['target'] = targets[0]['boxes'].detach().cpu().numpy()
                valid_predicts[image_id]['scores'] = preds[0]['scores'].detach().cpu().numpy()
            else:
                valid_predicts[image_id]['boxes'] = np.concatenate([valid_predicts[image_id]['boxes'], preds[0]['boxes'].detach().cpu().numpy()], axis=0)
                valid_predicts[image_id]['scores'] = np.concatenate([valid_predicts[image_id]['scores'], preds[0]['scores'].detach().cpu().numpy()], axis=0)
                sorted_idx = np.argsort(valid_predicts[image_id]['scores'])[::-1]
                valid_predicts[image_id]['boxes'] = valid_predicts[image_id]['boxes'][sorted_idx, :]
                valid_predicts[image_id]['scores'] = valid_predicts[image_id]['scores'][sorted_idx]
    

    # post processing
    for image_id in tqdm.tqdm(train_predicts.keys()):
        postprocessed_predict = postprocessing([train_predicts[image_id]], config["valid"])[0] if 'valid' in config.keys() else train_predicts[image_id][0]
        train_predicts[image_id]['processed_boxes'] = postprocessed_predict['boxes']
        train_predicts[image_id]['processed_scores'] = postprocessed_predict['scores']
    for image_id in tqdm.tqdm(valid_predicts.keys()):
        postprocessed_predict = postprocessing([valid_predicts[image_id]], config["valid"])[0] if 'valid' in config.keys() else valid_predicts[image_id][0]
        valid_predicts[image_id]['processed_boxes'] = postprocessed_predict['boxes']
        valid_predicts[image_id]['processed_scores'] = postprocessed_predict['scores']

    import pdb; pdb.set_trace()
    # 
    # processed_outputs = postprocessing(copy.deepcopy(preds), config["valid"]) if 'valid' in config.keys() else outputs


