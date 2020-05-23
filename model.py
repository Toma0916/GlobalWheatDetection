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
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from skimage.transform import AffineTransform, warp
import sklearn.metrics

# --- albumentations ---
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


def fasterrcnn_model(backbone, class_num=2, pool_layers_num=4, pooled_size=7, pretrained=True):
    """
    pool_layers_num: MultiScaleRoIAlignで使う層の数. 'resnet50_coco'では無視される. 安全そうな範囲で1~4で指定
    pooled_size: RoIPool後のmap size. 'resnet50_coco'では無視される. 安全そうな5~9で指定
    """

    backbone_list = {
        'resnet18': True,
        'resnet34': True,
        'resnet50': True,
        'resnet50_coco': True,  # いままでのやつ、headまでCOCOでpretrained
        'resnet101': True,  # batch_size=4は乗る
        'resnet152': True,   # batch_size=4は乗る
        'resnext50_32x4d': True,
        # 'resnext101_32x8d': True,  # エラー起きる
        # 'wide_resnet50_2': True,  # エラー起きる
        # 'wide_resnet101_2': True  # エラー起きる
    }

    assert backbone in backbone_list.keys(), 'Backbone\'s name is not valid. Available backbones: %s' % str(list(backbone_list.keys()))

    if backbone == 'resnet50_coco':
        # 今まで使っていたmodel、headまでpretrainedでweightsを読み込んでおり構造は弄れない
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features	
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)

    else:
        # backboneだけpretrained
        assert 1 <= pool_layers_num <= 4, 'pool_layers_num must be in [1, 2, 3, 4] You selected %d' % (pool_layers_num) 
        assert 5 <= pooled_size <= 9, 'pooled_size must be in [5, 6, 7, 8, 9] You selected %d' % (pooled_size) 

        # anchor_sizesはデフォルトから1スケール落とした。 default: ((32,), (64,), (128,), (256,), (512,))
        anchor_sizes = ((16), (32,), (64,), (128,), (256,))
        # anchor_ratiosは4:1の比を追加
        aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        # デフォルトでマルチスケールのRoIAlignになっている。headに近い4層から特徴を抽出しているはず
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[str(n) for n in range(pool_layers_num)], output_size=pooled_size, sampling_ratio=2)
        backbone = resnet_fpn_backbone(backbone, pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=class_num, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    return model 


    
def get_model(config):

    model_list = {
        'faster_rcnn': fasterrcnn_model,
    }

    assert config['name'] in model_list.keys(), 'Model\'s name is not valid. Available models: %s' % str(list(model_list.keys()))
    model = model_list[config['name']](**config['config'])
    return model