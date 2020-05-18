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


# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from skimage.transform import AffineTransform, warp
import sklearn.metrics

# --- albumentations ---
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


def fasterrcnn_model(backborn, class_num=2, pretrained=True):

    backborn_list = {
        'fasterrcnn_resnet50_fpn': torchvision.models.detection.fasterrcnn_resnet50_fpn
    }

    assert backborn in backborn_list.keys(), 'Backborn\'s name is not valid. Available backborns: %s' % str(list(backborn_list.keys()))
    model = backborn_list[backborn](pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)
    return model 



def get_model(config):

    model_list = {
        'faster_rcnn': fasterrcnn_model,
    }

    assert config['name'] in model_list.keys(), 'Model\'s name is not valid. Available models: %s' % str(list(model_list.keys()))
    model = model_list[config['name']](**config['config'])
    return model