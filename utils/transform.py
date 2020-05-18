import argparse
import os
from pathlib import Path
import random
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

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from skimage.transform import AffineTransform, warp
import sklearn.metrics

# --- albumentations ---
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch.transforms import ToTensorV2


class Transform:

    def __init__(self, config, is_train):

        if is_train or config['test_time_augment']:
            self.blur_p =  config['blur_p']  # blutをかける確率
            self.brightness_contrast_p =  config['brightness_contrast_p']  # brightness contrastを調整する確率
        else:
            self.blur_p = 0.0
            self.brightness_contrast_p = 0.0

    def __call__(self, example):

        image, target, image_id = example

        sample = {
            'image': image,
            'bboxes': target['boxes'],
            'labels': target['labels']
        }

        albumentation_transforms = A.Compose([
            ToTensorV2(p=1.0)  # convert numpy image to tensor
            ], 
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

        sample = albumentation_transforms(**sample)
        image = sample['image']
        target['boxes'] = torch.stack(tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)
        return image, target, image_id