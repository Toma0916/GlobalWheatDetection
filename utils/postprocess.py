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
import copy
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

    if threshold_score <= 0.0:
        return outputs
    
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

        # 処理によりboxが極端に減る場合は処理をスキップ
        if filtered_output['boxes'].shape[0] < 8:
            continue

        outputs[i] = filtered_output
    return outputs


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def non_maximum_supression_each(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def non_maximum_supression(outputs, threshold):
    
    for i, output in enumerate(outputs):

        picked_boxes, picked_scores = non_maximum_supression_each(output['boxes'], output['scores'], threshold)        
        processed_output = {}
        processed_output['boxes'] = np.array(picked_boxes)
        processed_output['scores'] = np.array(picked_scores)
        processed_output['labels'] = np.ones(len(picked_boxes))
        outputs[i] = processed_output

    return  outputs


def soft_non_maximum_supression(outputs, threshold):
    return outputs


def weighted_boxes_fusion(outputs, threshold):
    return outputs


def postprocessing(outputs, config):

    # detach and to cpu
    for i, output in enumerate(outputs):
        detached_output = {}
        detached_output['boxes'] = output['boxes'].cpu().detach().numpy()
        detached_output['labels'] = output['labels'].cpu().detach().numpy()
        detached_output['scores'] = output['scores'].cpu().detach().numpy()
        outputs[i] = detached_output

    # score filter 
    outputs = filter_score(copy.deepcopy(outputs), config['confidence_filter']['min_confidence'])

    # non maximamu supression
    if not 'non_maximum_supression' in config.keys():
        config['non_maximum_supression'] = {'apply': False}
    if config['non_maximum_supression']['apply']:
        outputs = non_maximum_supression(outputs, **config['non_maximum_supression']['config'])

    ensemble_boxes_method_list = {
        "nms": non_maximum_supression,
        "WIP_soft_nms": soft_non_maximum_supression,
        "WIP_wbf": weighted_boxes_fusion
    }
    
    if not "post_processor" in config.keys():
        return outputs
    
    ensemble_boxes_method_name = config['post_processor']['name'] 
    assert ensemble_boxes_method_name in ensemble_boxes_method_list.keys(), 'Ensembling boxes method\'s name is not valid. Available methods: %s' % str(list(ensemble_boxes_method_list.keys()))

    # non maximamu supression
    outputs = ensemble_boxes_method_list[ensemble_boxes_method_name](copy.deepcopy(outputs), **config['post_processor']['config'])
    return outputs

            