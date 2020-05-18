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

# # --- torch ---
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.nn import Sequential
# from torch.autograd import Variable
# from torch.utils.data.dataset import Dataset
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# # --- models ---
# from sklearn import preprocessing
# from sklearn.model_selection import KFold, train_test_split
# from skimage.transform import AffineTransform, warp
# import sklearn.metrics

# # --- albumentations ---
# import albumentations as A
# from albumentations.core.transforms_interface import DualTransform


def convert_dataframe(dataframe):
    """
    convert dataframe format
    bbox -> x, y, w, h
    """
    dataframe['x'] = -1
    dataframe['y'] = -1
    dataframe['w'] = -1
    dataframe['h'] = -1

    def expand_bbox(x):
        r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
        if len(r) == 0:
            r = [-1, -1, -1, -1]
        return r

    dataframe[['x', 'y', 'w', 'h']] = np.stack(dataframe['bbox'].apply(lambda x: expand_bbox(x)))
    dataframe.drop(columns=['bbox'], inplace=True)
    dataframe['x'] = dataframe['x'].astype(np.float)
    dataframe['y'] = dataframe['y'].astype(np.float)
    dataframe['w'] = dataframe['w'].astype(np.float)
    dataframe['h'] = dataframe['h'].astype(np.float)
    return dataframe


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def filter_large_bboxes(targets, config):
    if 'bboxes_size_thres' not in config['general'].keys():
        return targets
    thres = config['general']['bboxes_size_thres']
    targets_filtered = []
    for target in targets:
        target_filtered = {'boxes': target['boxes'][target['area']<thres], 
                           'labels': target['labels'][target['area']<thres],
                           'area': target['area'][target['area']<thres],
                           'image_id':target['image_id']}
        targets_filtered.append(target_filtered)
        if (target['labels'].shape[0] != target_filtered['labels'].shape[0]) and config['debug']:
            print("OHH!!!! DETECTED LARGE BBOX!!!!")
    return targets_filtered