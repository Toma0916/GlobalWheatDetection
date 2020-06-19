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
from utils.functions import convert_dataframe

# detect and make pseudo label for test set by using trained model
def prepare_pseudo_labels(loaded_models, valid_data_loader, config):

    if config['apply'] is False:
        return loaded_models

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
                
        loaded_models[key]['pseudo_df'] = pd.DataFrame(test_pseudo)
    return loaded_models