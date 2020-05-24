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

from utils.functions import detach_outputs


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
# -> aranged
# --------------------------------------------------------
def non_maximum_supression_each(bounding_boxes, confidence_score, threshold=None, sigma=None, method_type='original'):
    """
    When method type is 'original, threshold is used and sigma is ignored
    When method type is 'soft, threshold is ignored and sigma is used
    """

    assert method_type in ['original', 'soft'], "method_type must be in ['original', 'soft']"

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
        if method_type == 'original':
            left = np.where(ratio < threshold)
            order = order[left]
        elif method_type == 'soft':
            weights = np.exp(-(ratio*ratio)/sigma)
            # 自分のorderより小さいところ？？に修正
            confidence_score[:(order.shape[0]-1)] = weights * confidence_score[:(order.shape[0]-1)]
            order = order[:-1]

    return picked_boxes, picked_score


def non_maximum_supression(outputs, threshold):
    
    for i, output in enumerate(outputs):

        picked_boxes, picked_scores = non_maximum_supression_each(output['boxes'], output['scores'], threshold=threshold, sigma=None, method_type='original')        
        processed_output = {}
        processed_output['boxes'] = np.array(picked_boxes)
        processed_output['scores'] = np.array(picked_scores)
        processed_output['labels'] = np.ones(len(picked_boxes))
        outputs[i] = processed_output

    return  outputs


def soft_non_maximum_supression(outputs, sigma):

    for i, output in enumerate(outputs):

        picked_boxes, picked_scores = non_maximum_supression_each(output['boxes'], output['scores'], threshold=None, sigma=sigma, method_type='soft')    
        processed_output = {}
        processed_output['boxes'] = np.array(picked_boxes)
        processed_output['scores'] = np.array(picked_scores)
        processed_output['labels'] = np.ones(len(picked_boxes))
        outputs[i] = processed_output

    return outputs



def weighted_boxes_fusion_each(bounding_boxes, confidence_score, threshold=None):
    """
    When method type is 'original, threshold is used and sigma is ignored
    When method type is 'soft, threshold is ignored and sigma is used
    """

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

    # Onetime variables
    fusion_boxes = np.array([])
    material_boxes = []
    material_confidences = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Initialize
    fusion_boxes = boxes[order[-1]:order[-1]+1, :]
    material_boxes = [boxes[order[-1]:order[-1]+1, :]]
    material_confidences = [score[order[-1]:order[-1]+1]]
    order = order[:-1]

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        f_start_x = fusion_boxes[:, 0]
        f_start_y = fusion_boxes[:, 1]
        f_end_x = fusion_boxes[:, 2]
        f_end_y = fusion_boxes[:, 3]
        f_areas = (f_end_x - f_start_x + 1) * (f_end_y - f_start_y + 1)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], f_start_x)
        x2 = np.minimum(end_x[index], f_start_y)
        y1 = np.maximum(start_y[index], f_end_x)
        y2 = np.minimum(end_y[index], f_end_y)

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + f_areas - intersection)

        # Count match
        mtcs = np.where(threshold < ratio)[0]

        if mtcs.shape[0] == 0:
            fusion_boxes = np.append(fusion_boxes, boxes[index:index+1, :], axis=0)
            material_boxes.append(boxes[index:index+1, :])
            material_confidences.append(score[index:index+1])
            order = order[:-1]
            continue
        
        for mtc in mtcs:
            fusion_boxes = np.append(fusion_boxes, boxes[index:index+1, :], axis=0)
            material_boxes[mtc] = np.append(material_boxes[mtc], boxes[index:index+1], axis=0)
            material_confidences[mtc] = np.append(material_confidences[mtc], score[index:index+1])



        order = order[:-1]

    # Fusion boxes
    for i in range(len(material_boxes)):

        material_num = material_boxes[i].shape[0]
        cofidence_sum = np.sum(material_confidences[i])

        fusion_box = np.zeros(4)
        fusion_box[0] = np.sum(material_boxes[i][:, 0] * material_confidences[i])/cofidence_sum
        fusion_box[1] = np.sum(material_boxes[i][:, 1] * material_confidences[i])/cofidence_sum
        fusion_box[2] = np.sum(material_boxes[i][:, 2] * material_confidences[i])/cofidence_sum
        fusion_box[3] = np.sum(material_boxes[i][:, 3] * material_confidences[i])/cofidence_sum

        fusion_confidence = cofidence_sum / material_num

        picked_boxes.append(fusion_box)
        picked_score.append(fusion_confidence)
  
    return picked_boxes, picked_score


def weighted_boxes_fusion(outputs, threshold):

    for i, output in enumerate(outputs):

        picked_boxes, picked_scores = weighted_boxes_fusion_each(output['boxes'], output['scores'], threshold)    
        processed_output = {}
        processed_output['boxes'] = np.array(picked_boxes)
        processed_output['scores'] = np.array(picked_scores)
        processed_output['labels'] = np.ones(len(picked_boxes))
        outputs[i] = processed_output
    return outputs


def postprocessing(outputs, config):

    # detach and to cpu
    outputs = detach_outputs(outputs)
            
    if not config["post_processor"]["name"] in config.keys():    
        ensemble_boxes_method_list = {
            "nms": non_maximum_supression,
            "soft_nms": soft_non_maximum_supression,
            "wbf": weighted_boxes_fusion
        }
        ensemble_boxes_method_name = config['post_processor']['name'] 
        assert ensemble_boxes_method_name in ensemble_boxes_method_list.keys(), 'Ensembling boxes method\'s name is not valid. Available methods: %s' % str(list(ensemble_boxes_method_list.keys()))
        outputs = ensemble_boxes_method_list[ensemble_boxes_method_name](copy.deepcopy(outputs), **config['post_processor']['config'])

    # score filter 
    outputs = filter_score(copy.deepcopy(outputs), config['confidence_filter']['min_confidence'])
    
    return outputs

            