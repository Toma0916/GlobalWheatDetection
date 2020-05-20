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


def postprocessing(outputs, config):

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

    # import pdb; pdb.set_trace()

    # non maximamu supression
    pass

    return outputs