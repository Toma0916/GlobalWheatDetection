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
    """
    https://github.com/katsura-jp/tour-of-albumentations 参考
    """

    def __init__(self, config, is_train):

        if is_train or config['test_time_augment']:

            # flip系
            self.horizontal_frip = config['horizontal_frip'] if ('horizontal_frip' in config) else {'p': 0.0}
            self.vertical_frip = config['vertical_frip'] if ('vertical_frip' in config) else {'p': 0.0}

            # blur系
            self.blur = config['blur'] if ('blur' in config) else {'p': 0.0}
            self.motion_blur = config['motion_blur'] if ('motion_blur' in config) else {'p': 0.0}
            self.median_blur = config['median_blur'] if ('median_blur' in config) else {'p': 0.0}
            self.gaussian_blur = config['gaussian_blur'] if ('gaussian_blur' in config) else {'p': 0.0}

            # distortion系
            self.optical_distorion = config['optical_distorion'] if ('optical_distorion' in config) else {'p': 0.0}
            self.grid_distorion = config['grid_distorion'] if ('grid_distorion' in config) else {'p': 0.0}
            self.elastic_distorion = config['elastic_distorion'] if ('elastic_distorion' in config) else {'p': 0.0}

            # color系
            self.clahe = config['clahe'] if ('clahe' in config) else {'p': 0.0}
            self.chennel_shufflle = config['chennel_shufflle'] if ('chennel_shufflle' in config) else {'p': 0.0} 
            self.random_gamma = config['random_gamma'] if ('random_gamma' in config) else {'p': 0.0}
            self.hue_saturation_value = config['hue_saturation_value'] if ('hue_saturation_value' in config) else {'p': 0.0}
            self.rgb_shift = config['rgb_shift'] if ('rgb_shift' in config) else {'p': 0.0}
            self.random_brightness = config['random_brightness'] if ('random_brightness' in config) else {'p': 0.0}
            self.random_contrast = config['random_contrast'] if ('random_contrast' in config) else {'p': 0.0}
            
            # noise系
            self.gaussian_noise = config['gaussian_noise'] if ('gaussian_noise' in config) else {'p': 0.0}
            self.cutout = config['cutout'] if ('cutout' in config) else {'p': 0.0}


        else:
            self.horizontal_frip = {'p': 0.0}
            self.vertical_frip = {'p': 0.0} 
            self.blur = {'p': 0.0}
            self.motion_blur = {'p': 0.0}
            self.median_blur = {'p': 0.0}
            self.gaussian_blur = {'p': 0.0}
            self.optical_distorion = {'p': 0.0}
            self.grid_distorion = {'p': 0.0}
            self.elastic_distorion = {'p': 0.0}
            self.clahe = {'p': 0.0}
            self.chennel_shufflle = {'p': 0.0}
            self.random_gamma = {'p': 0.0}
            self.hue_saturation_value = {'p': 0.0}
            self.rgb_shift = {'p': 0.0}
            self.random_brightness = {'p': 0.0}
            self.random_contrast = {'p': 0.0}
            self.gaussian_noise = {'p': 0.0}
            self.cutout = {'p': 0.0}

    def __call__(self, example):

        image, target, image_id = example

        sample = {
            'image': image,
            'bboxes': target['boxes'],
            'labels': target['labels']
        }

        albumentation_transforms = A.Compose([
            A.HorizontalFlip(**self.horizontal_frip),
            A.VerticalFlip(**self.vertical_frip),
            A.Blur(**self.blur),
            A.MotionBlur(**self.motion_blur),
            A.MedianBlur(**self.median_blur),
            A.GaussianBlur(**self.gaussian_blur),
            A.OpticalDistortion(**self.optical_distorion),
            A.GridDistortion(**self.grid_distorion),
            A.ElasticTransform(**self.elastic_distorion),
            A.CLAHE(**self.clahe),
            A.ChannelShuffle(**self.chennel_shufflle),
            A.RandomGamma(**self.random_gamma),
            A.HueSaturationValue(**self.hue_saturation_value),
            A.RGBShift(**self.rgb_shift),
            A.RandomBrightness(**self.random_brightness),
            A.RandomContrast(**self.random_contrast),
            A.GaussNoise(**self.gaussian_noise),
            A.Cutout(**self.cutout),
            ToTensorV2(p=1.0)  # convert numpy image to tensor
            ], 
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

        sample = albumentation_transforms(**sample)
        image = sample['image']
        target['boxes'] = torch.stack(tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)
        return image, target, image_id