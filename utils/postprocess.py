import argparse
import os
from pathlib import Path
import random
import math
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

### Define custom lr schedulers. Might be better to make another file like 'utils/schedulers.py' for the classes below.
# from: https://github.com/lyakaap/pytorch-template/blob/master/src/lr_scheduler.py
from bisect import bisect_right

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


def filter_score(outputs, threshold_score):
    
    for i, output in enumerate(outputs):
        filtered_output = {key: [] for key in outputs[i].keys()}
        scores = output['scores']
        for j, score in enumerate(scores):
            if threshold_score <= score:
                filtered_output['boxes'].append(output['boxes'][j])
                filtered_output['labels'].append(output['labels'][j])
                filtered_output['scores'].append(output['scores'][j])
        filtered_output['boxes'] = np.array(filtered_output['boxes'])
        filtered_output['labels'] = np.array(filtered_output['labels'])
        filtered_output['scores'] = np.array(filtered_output['scores'])

        outputs[i] = filtered_output
    return outputs


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def non_maximum_supression_each(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def non_maximum_supression(outputs, threshold):
    
    for i, output in enumerate(outputs):

        keep_index = non_maximum_supression_each(np.concatenate([output['boxes'], np.reshape(output['scores'], (-1, 1))], axis=1), threshold)
        
        processed_output = {}
        processed_output['boxes'] = output['boxes'][keep_index]
        processed_output['labels'] = output['labels'][keep_index]
        processed_output['scores'] = output['scores'][keep_index]
        outputs[i] = processed_output
    return  outputs

def soft_nms(outputs, threshold):
    return outputs

def weighted_boxes_fusion(outputs, threshold):
    return outputs

def postprocessing(outputs, config):
    ensemble_boxes_method_list = {
        "nms": non_maximum_supression,
        "WIP_soft_nms": soft_nms,
        "WIP_wbf": weighted_boxes_fusion
    }
    # detach and to cpu
    for i, output in enumerate(outputs):
        detached_output = {}
        detached_output['boxes'] = output['boxes'].cpu().detach().numpy()
        detached_output['labels'] = output['labels'].cpu().detach().numpy()
        detached_output['scores'] = output['scores'].cpu().detach().numpy()
        outputs[i] = detached_output

    # score filter 
    threshold_score = config['confidence_filter']['min_confidence']
    if 0 < threshold_score:
        outputs = filter_score(outputs, threshold_score)

    if not "ensemble_boxes_method" in config.keys():
        return outputs
    ensemble_boxes_method_name = config['ensemble_boxes_method']['name'] 
    assert ensemble_boxes_method_name in ensemble_boxes_method_list.keys(), 'Ensembling boxes method\'s name is not valid. Available methods: %s' % str(list(optimizer_list.keys()))
    # non maximamu supression
    outputs = ensemble_boxes_method_list[ensemble_boxes_method_name](outputs, **config['ensemble_boxes_method']['config'])
    return outputs