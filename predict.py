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
from utils.functions import get_config, convert_dataframe, format_config_by_baseconfig, randomname
from utils.metric import calculate_score_for_each
from utils.postprocess import postprocessing
from utils.optimize import PostProcessOptimizer
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


# load model and predict
def predict_original_for_loader(loaded_models, dataloader):
    
    predicts = defaultdict(lambda: {'original': defaultdict(dict), 'target': defaultdict(dict), 'processed': defaultdict(dict)})
    metrics = []

    for i, key in enumerate(loaded_models.keys()):
        print('【%d/%d】' % (i+1, len(loaded_models.keys())))
        model = loaded_models[key]['model']
        for images, targets, image_ids in tqdm.tqdm(dataloader):
            image_id = image_ids[0]
            preds, loss_dict = model(images, targets)
            metrics.append(calculate_score_for_each(preds, targets))

            if 'boxes' not in predicts[image_id]['original']:
                predicts[image_id]['original']['boxes'] = preds[0]['boxes']
                predicts[image_id]['original']['scores'] = preds[0]['scores']
                predicts[image_id]['target']['boxes'] = targets[0]['boxes']
            else:
                predicts[image_id]['original']['boxes'] = np.concatenate([predicts[image_id]['original']['boxes'], preds[0]['boxes']], axis=0)
                predicts[image_id]['original']['scores'] = np.concatenate([predicts[image_id]['original']['scores'], preds[0]['scores']], axis=0)
                sorted_idx = np.argsort(predicts[image_id]['original']['scores'])[::-1]
                predicts[image_id]['original']['boxes'] = predicts[image_id]['original']['boxes'][sorted_idx, :]
                predicts[image_id]['original']['scores'] = predicts[image_id]['original']['scores'][sorted_idx]
    return predicts, np.array(metrics)
    

def predict_original(loaded_models, train_data_loader, valid_data_loader):
    print('predicting train...')
    train_predicts, train_metrics = predict_original_for_loader(loaded_models, train_data_loader)
    print('predicting valid...')
    valid_predicts, valid_metrics = predict_original_for_loader(loaded_models, valid_data_loader)
    return train_predicts, train_metrics, valid_predicts, valid_metrics
    
    
def apply_postprocess(train_predicts, valid_predicts, config):

        train_predicts = copy.deepcopy(train_predicts)
        valid_predicts = copy.deepcopy(valid_predicts)

        train_metrics = []
        valid_metrics = []
        for image_id in train_predicts.keys():
            processed = postprocessing([train_predicts[image_id]['original']], config)
            metrics = calculate_score_for_each(processed, [train_predicts[image_id]['target']])
            train_metrics.append(metrics)
            train_predicts[image_id]['processed']['boxes'] = processed[0]['boxes']
            train_predicts[image_id]['processed']['scores'] = processed[0]['scores']
        
        for image_id in valid_predicts.keys():
            processed = postprocessing([valid_predicts[image_id]['original']], config)
            metrics = calculate_score_for_each(processed, [valid_predicts[image_id]['target']])
            valid_metrics.append(metrics)
            valid_predicts[image_id]['processed']['boxes'] = processed[0]['boxes']
            valid_predicts[image_id]['processed']['scores'] = processed[0]['scores']
        
        train_metrics = np.array(train_metrics)
        valid_metrics = np.array(valid_metrics)

        return train_predicts, train_metrics, valid_predicts, valid_metrics

if __name__ == '__main__':

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('json_path')
    args = parser.parse_args()
    device = torch.device('cuda:0')

    assert os.path.exists(args.json_path), "json\'s name '%s' is not valid." % args.json_path

    with open(args.json_path, 'r') as f:
        predict_config = json.load(f)
    debug = predict_config['debug']
    model_paths = predict_config['model_paths']

    SRC_DIR = Path('.').resolve()/'src'
    TRAIN_IMAGE_DIR = SRC_DIR/'train'
    TEST_IMAGE_DIR= SRC_DIR/'test'
    DATAFRAME = convert_dataframe(pd.read_csv(str(SRC_DIR/'train.csv')))    
    OUTPUT_DIR = Path('.').resolve()/'output'

    loaded_models = defaultdict(dict)
    for path in model_paths:
        config_path = OUTPUT_DIR/path/'config.json'
        with open(str(config_path), 'r') as f:
            config = json.load(f)

        if 'kfold' in config['general'] and 0 < config['general']['kfold']:
            print('[WARN]: not supporting k-fold. Execute sys.exit().')
            sys.exit()  # [WIP]: reload weights
        
        weight_path = list(sorted((OUTPUT_DIR/path).glob('*.pt')))[-1]  # latest weight
        loaded_models[path]['config'] = config
        loaded_models[path]['weight_path'] = weight_path
        model = Model(loaded_models[path]['config']['model'])
        model = model.load_state_dict(str(loaded_models[path]['weight_path'])).to(device)
        model.eval()
        loaded_models[path]['model'] = model
    
    # check random_seed and train_valid_split
    assert sanity_check(loaded_models), 'The models you selected are invalid.'
    
    # set seed (not enough for complete reproducibility)
    random_seed = loaded_models[model_paths[0]]['config']['general']['seed']
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  
    torch.cuda.manual_seed(random_seed) 
    
    def worker_init_fn(worker_id):   
        random.seed(worker_id+random_seed)   
        np.random.seed(worker_id+random_seed) 

    train_ids, valid_ids = train_valid_split(DATAFRAME, loaded_models[model_paths[0]]['config'])[0]
    train_ids = train_ids[:10] if debug else train_ids
    valid_ids = valid_ids[:2] if debug else valid_ids
    train_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(train_ids), :]
    valid_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(valid_ids), :]
    train_dataset = GWDDataset(train_dataframe, TRAIN_IMAGE_DIR, loaded_models[model_paths[0]]['config'], is_train=True, do_transform=False)
    valid_dataset = GWDDataset(valid_dataframe, TRAIN_IMAGE_DIR, loaded_models[model_paths[0]]['config'], is_train=False, do_transform=False)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, collate_fn=collate_fn)    
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, collate_fn=collate_fn)

    # predict train and valid 
    # ensemble if you selected multiple models
    train_predicts, original_train_metrics, valid_predicts, original_valid_metrics = predict_original(loaded_models, train_data_loader, valid_data_loader)

    # optimize postprocess
    postprocess_optimizer = PostProcessOptimizer(train_predicts, valid_predicts)
    best_nms_threshold, best_nms_min_confidence = postprocess_optimizer.optimize_nms(n_calls=50)
    best_soft_nms_sigma, best_soft_nms_min_confidence = postprocess_optimizer.optimize_soft_nms(n_calls=50)
    best_wbf_threshold, best_wbf_min_confidence = postprocess_optimizer.optimize_wbf(n_calls=50)

    nms_config = get_config(name='nms', threshold=best_nms_threshold, min_confidence=best_nms_min_confidence)
    soft_nms_config = get_config(name='soft_nms', sigma=best_soft_nms_sigma, min_confidence=best_soft_nms_min_confidence)
    wbf_config = get_config(name='wbf', threshold=best_wbf_threshold, min_confidence=best_wbf_min_confidence)
    nms_train_predicts, nms_train_metrics, nms_valid_predicts, nms_valid_metrics = apply_postprocess(train_predicts, valid_predicts, nms_config)
    soft_nms_train_predicts, soft_nms_train_metrics, soft_nms_valid_predicts, soft_nms_valid_metrics = apply_postprocess(train_predicts, valid_predicts, soft_nms_config)
    wbf_train_predicts, wbf_train_metrics, wbf_valid_predicts, wbf_valid_metrics = apply_postprocess(train_predicts, valid_predicts, wbf_config)
    
    print('original train metrics: ', np.mean(original_train_metrics))
    print('original valid metrics: ', np.mean(original_valid_metrics))
    print('nms train metrics: ', np.mean(nms_train_metrics))
    print('nms valid metrics: ', np.mean(nms_valid_metrics))
    print('soft nms train metrics: ', np.mean(soft_nms_train_metrics))
    print('soft nms valid metrics: ', np.mean(soft_nms_valid_metrics))
    print('wbf train metrics: ', np.mean(wbf_train_metrics))
    print('wbf valid metrics: ', np.mean(wbf_valid_metrics))

    # # ここでサンプル描画
    # images, targets, image_ids = iter(train_data_loader).next()
    # image = (cv2.UMat(np.transpose(images[0].detach().cpu().numpy(), (1, 2, 0))).get() * 255).astype(np.uint8)
    # image_id = image_ids[0]
    # target_box = train_predicts[image_ids[0]]['target']['boxes'].detach().cpu().numpy()
    # for j in range(target_box.shape[0]):
    #     box = target_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
    
    # original_box = train_predicts[image_ids[0]]['original']['boxes']
    # for j in range(original_box.shape[0]):
    #     box = original_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 220, 0), 1)
    
    # processed_box = train_predicts[image_ids[0]]['processed']['boxes']
    # for j in range(processed_box.shape[0]):
    #     box = processed_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 3)
    # cv2.imwrite('./sample.png', image)

    # images, targets, image_ids = iter(valid_data_loader).next()
    # image = (cv2.UMat(np.transpose(images[0].detach().cpu().numpy(), (1, 2, 0))).get() * 255).astype(np.uint8)
    # image_id = image_ids[0]
    # target_box = valid_predicts[image_ids[0]]['target']['boxes'].detach().cpu().numpy()
    # for j in range(target_box.shape[0]):
    #     box = target_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
    
    # original_box = valid_predicts[image_ids[0]]['original']['boxes']
    # for j in range(original_box.shape[0]):
    #     box = original_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 220, 0), 1)
    
    # processed_box = valid_predicts[image_ids[0]]['processed']['boxes']
    # for j in range(processed_box.shape[0]):
    #     box = processed_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 3)
    # cv2.imwrite('./sample2.png', image)

    # import pdb; pdb.set_trace()


