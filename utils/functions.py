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
import string
import copy

import numpy as np 
from numpy.random.mtrand import RandomState
import pandas as pd 

from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import cv2

import matplotlib.pyplot as plt


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


def get_filtered_bboxes(target, threshold, max_or_min):
    """
    if max: max_or_min = 1
    if min: max_or_min = -1   
    """
    filtered = ((max_or_min * target['area']) < (max_or_min * threshold)).numpy()
    target_filtered = {'boxes': target['boxes'][filtered], 
                       'labels': target['labels'][filtered],
                       'area': target['area'][filtered],
                       'iscrowd': target['iscrowd'][filtered],
                       'image_id': target['image_id']}
    return target_filtered


def filter_bboxes_by_size(target, config):

    if config is None:
        return target

    if config['max_bbox_size'] >= 0:
        target = get_filtered_bboxes(target, config['max_bbox_size'], 1)
    if config['min_bbox_size'] >= 0:
        target = get_filtered_bboxes(target, config['min_bbox_size'], -1)
    return target


def drop_bboxes_by_probability(boxes, p):

    if p <= 0.0:
        return boxes

    dropped_boxes = boxes[(p < np.random.rand(boxes.shape[0])), :]

    if dropped_boxes.shape[0] == 0:
        return boxes
    return dropped_boxes


def vibrate_bboxes_with_ratio(boxes, ratio, image_size): 

    if ratio <= 0:
        return boxes

    h, w = image_size

    boxes_width = boxes[:, 2] - boxes[:, 0]
    boxes_height = boxes[:, 3] - boxes[:, 1]

    for i in range(boxes.shape[0]):
        boxes[i][0] = np.max([0, boxes[i][0] + int(ratio * np.random.randint(boxes_width[i]*(-1), boxes_width[i]))])
        boxes[i][1] = np.max([0, boxes[i][1] + int(ratio * np.random.randint(boxes_height[i]*(-1), boxes_height[i]))])
        boxes[i][2] = np.min([w, boxes[i][2] + int(ratio * np.random.randint(boxes_width[i]*(-1), boxes_width[i]))])
        boxes[i][3] = np.min([h, boxes[i][3] + int(ratio * np.random.randint(boxes_height[i]*(-1), boxes_height[i]))])
    
    return boxes

def dict_flatten_old(target, separator='_'):
    if not isinstance(target, dict):
        raise ValueError
    
    if not any(filter(lambda x: isinstance(x, dict), target.values())):
        return target
    
    dct = {}
    for key, value in target.items():
        if key == 'config':
            continue
        if isinstance(value, dict):
            for k, v in dict_flatten_old(value, separator).items():
                if type(v) is list:
                    v = ' '.join(sorted(v))
                dct[str(key) + separator + str(k)] = v
        else:
            if type(value) is list:
                value = ' '.join(sorted(value))
            dct[key] = value
            
    return dct

def dict_flatten(target, target_base, separator='_'):
    if not isinstance(target, dict):
        raise ValueError
    
    dct = {}
    idx = 0
    for key, value in target.items():
        if key == 'config' or key == 'prefix':
            continue
        if isinstance(value, dict):
            prfx = target_base[key]['prefix']+separator if 'prefix' in target_base[key].keys() else ''
            if 'p' in value.keys():
                dct[prfx + key] = value['p']
            else:
                for k, v in dict_flatten(value, target_base[key], separator).items():
                    dct[prfx + k] = v
        else:
            if key in target_base.keys():
                dct[target_base[key][1]] = value
            else:
                dct[key] = value
        idx += 1
    return dct

def params_to_mlflow_format(d, base_config_path='./sample_json/BASE_CONFIG.json'):
    with open(base_config_path, 'r') as f:
        d_base = json.load(f)
    return dict_flatten(d, d_base)

def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)


def func(d_base, d):
    for k, v in d_base.items():
        if isinstance(v, dict):
            if not k in d.keys():
                d[k] = {}
            if k != 'config' and k != 'p':
                d[k] = func(d_base[k], d[k])
        elif not k in d.keys():
            if isinstance(v, list):
                d[k] = d_base[k][0]
            else:
                d[k] = d_base[k]
    return d

  
def format_config_by_baseconfig(config, base_config_path='./sample_json/BASE_CONFIG.json'):
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    return func(copy.deepcopy(base_config), copy.deepcopy(config))



def detach_outputs(outputs):

    if type(outputs[0]['boxes']) is np.ndarray:
        return outputs

    # detach and to cpu
    for i, output in enumerate(outputs):
        detached_output = {}
        detached_output['boxes'] = output['boxes'].cpu().detach().numpy()
        detached_output['labels'] = output['labels'].cpu().detach().numpy()
        detached_output['scores'] = output['scores'].cpu().detach().numpy()
        outputs[i] = detached_output
    return outputs

def xyxy2xywh(boxes):
    return 0

def xywh2xyxy(boxes):
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    return boxes

def random_box(h, w, l):
    cut_rat = np.sqrt(1. - l)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w//2 - w//4, w//2 + w//4)
    cy = np.random.randint(h//2 - h//4, h//2 + h//4)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    if bbx1 == 0:
      bbx2 = np.clip(bbx2, w // 6, None)
    elif bbx2 == w:
      bbx1 = np.clip(bbx1, None, w - w // 6)
    
    if bby1 == 0:
      bby2 = np.clip(bby2, h // 6, None)
    elif bby2 == h:
      bby1 = np.clip(bby1, None, h - h // 6)

      
    return bbx1, bby1, bbx2, bby2


def calc_box_overlap(boxes, box):
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    area = (end_x - start_x + 1) * (end_y - start_y + 1)
    x1 = np.maximum(box[0], start_x)
    x2 = np.minimum(box[2], end_x)
    y1 = np.maximum(box[1], start_y)
    y2 = np.minimum(box[3], end_y)
    w = np.maximum(0.0, x2 - x1 + 1)
    h = np.maximum(0.0, y2 - y1 + 1)
    intersection = w * h
    ratio = intersection/area
    return ratio