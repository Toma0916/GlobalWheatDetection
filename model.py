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


def fasterrcnn_model(backbone, class_num=2, pretrained=True):

    backbone_list = {
        'resnet18': True,
        'resnet34': True,
        'resnet50': True,
        'resnet101': True,  # batch_size=4は乗る
        'resnet152': True,   # batch_size=4は乗る
        'resnext50_32x4d': True,
        # 'resnext101_32x8d': True,  # エラー起きる
        # 'wide_resnet50_2': True,  # エラー起きる
        # 'wide_resnet101_2': True  # エラー起きる
    }

    assert backbone in backbone_list.keys(), 'Backbone\'s name is not valid. Available backbones: %s' % str(list(backbone_list.keys()))

    backbone = resnet_fpn_backbone(backbone, pretrained=pretrained)

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes=class_num, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    return model 


    
def get_model(config):

    model_list = {
        'faster_rcnn': fasterrcnn_model,
    }

    assert config['name'] in model_list.keys(), 'Model\'s name is not valid. Available models: %s' % str(list(model_list.keys()))
    model = model_list[config['name']](**config['config'])
    return model