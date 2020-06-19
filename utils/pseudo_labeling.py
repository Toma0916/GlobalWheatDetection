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
import itertools


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
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from torch.utils.data.dataset import Dataset

from optimizer import get_optimizer
from scheduler import get_scheduler

from utils.functions import convert_dataframe
from utils.dataset import GWDDataset, collate_fn
from train import train_epoch

# detect and make pseudo label for test set by using trained model
def prepare_pseudo_labels(loaded_models, valid_data_loader, config):

    if config['apply'] is False:
        return

    for key in loaded_models.keys():

        test_pseudo = []

        model = loaded_models[key]['model']
        for images, targets, image_ids in tqdm.tqdm(valid_data_loader):
            image_id = image_ids[0]
            preds, _ = model(images, targets)
            pred = preds[0]
            boxes = pred['boxes'].detach().cpu().numpy()[np.where(config['detection_threshold'] < pred['scores'].detach().cpu().numpy())]
            shape = images[0].detach().cpu().numpy().shape
            width = shape[2]
            height = shape[1]
            boxes = boxes.astype(np.int)
            boxes[:, 0] = np.clip(boxes[:, 0], 0, width-1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, height-1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, width-1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, height-1)
            if boxes.shape[0] == 0:
                continue
            else:
                for box in boxes:
                    test_pseudo.append({
                        'image_id': image_id,
                        'width': width,
                        'height': height,
                        'source': 'pseudo',
                        'x': box[0],
                        'y': box[1],
                        'w': box[2] - box[0],  # model output format is pascal voc, convert to coco
                        'h': box[3] - box[1]
                    })
                
        loaded_models[key]['pseudo_dataframe'] = pd.DataFrame(test_pseudo)
    return

    
def retrain_with_pseudo_label(loaded_models, train_ids, valid_ids, TRAIN_IMAGE_DIR, DATAFRAME, config):

    if 'pseudo_dataframe' not in loaded_models[list(loaded_models.keys())[0]]:
        return
    
    def worker_init_fn(worker_id):   
        random.seed(worker_id+random_seed)   
        np.random.seed(worker_id+random_seed) 

    for key in loaded_models.keys():    

        # make dataloader with pseudo label
        model_config = loaded_models[key]['config']
        dataframe_with_pseudo = pd.concat([DATAFRAME.loc[DATAFRAME['image_id'].isin(train_ids), :], loaded_models[key]['pseudo_dataframe']], axis=0)
        retrain_dataset = GWDDataset(dataframe_with_pseudo, TRAIN_IMAGE_DIR, model_config, is_train=True, do_transform=False)
        # dataset for retrain
        retrain_data_loader = DataLoader(retrain_dataset, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, collate_fn=collate_fn)    

        model = copy.deepcopy(loaded_models[key]['model'])
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = get_optimizer(model_config['train']['optimizer'], trainable_params)

        # retraining
        print("Retraining %s" % key)
        for epoch in range(0, config['epochs']):
            if model_config['general']['kfold'] < 0:
                print("\r[Epoch %d]" % epoch)
            train_epoch(model, retrain_data_loader, None, optimizer)
        model.eval()
        loaded_models[key]['pseudo_model'] = model
    return 




