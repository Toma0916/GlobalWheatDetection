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



def moisac(image_list, target_list, image_id_list):

    s = 1024
    h = 1024
    w = 1024

    boxes4 = []
    joined_image_id = '_'.join(image_id_list)

    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y

    for i in range(len(image_list)):

        img = image_list[i]               
        image_id = image_id_list[i]

        # Load image
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114.0/255.0, dtype=np.float32)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # calculate coco bounding box
        boxes_pascalvoc = target_list[i]['boxes']
        boxes_coco = []
        for box in boxes_pascalvoc:
            b_x1, b_y1, b_x2, b_y2 = box
            b_xc, b_yc, b_w, b_h = 0.5*b_x1/s+0.5*b_x2/s, 0.5*b_y1/s+0.5*b_y2/s, abs(b_x2/s-b_x1/s), abs(b_y2/s-b_y1/s)
            boxes_coco.append([b_xc, b_yc, b_w, b_h])
        boxes_coco = np.array(boxes_coco)

        x = boxes_coco
        boxes = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            boxes[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
            boxes[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
            boxes[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
            boxes[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
        boxes4.append(boxes)

        
    
    # Concat/clip labels
    if len(boxes4):
        boxes4 = np.concatenate(boxes4, 0)
        boxes4 = np.clip(boxes4, 0, 2 * s) 
        boxes4 = np.array([box/2. for box in boxes4 if ((box[0]+1 < box[2]) and (box[1]+1 < box[3]))])  # resize and remove outliers
        img4 = cv2.resize(img4, (s, s), interpolation=cv2.INTER_LINEAR)  # resize image by `INTER_LINEAR`
        return img4, boxes4, joined_image_id
    else:
        # return original set if the number of bounding boxese after mosaic process is zero
        return image_list[0], target_list[0], image_id_list[0]

    


class Transform:
    """
    https://github.com/katsura-jp/tour-of-albumentations 参考
    """

    def __init__(self, all_config, is_train):
        self.model_name = all_config['model']['name']
        self.img_size = all_config['model']['config']['image_size'] if self.model_name=='efficient_det' else 1024
        config = all_config['train']['augment']
        if is_train or config['test_time_augment']:

            # mosaic
            self.mosaic = config['mosaic'] if ('mosaic' in config) else {'p': 0.0}

            # flip系
            self.horizontal_flip = config['horizontal_flip'] if ('horizontal_flip' in config) else {'p': 0.0}
            self.vertical_flip = config['vertical_flip'] if ('vertical_flip' in config) else {'p': 0.0}
            self.random_rotate_90 = config['random_rotate_90'] if ('random_rotate_90' in config) else {'p': 0.0}

            # crop系
            self.random_sized_bbox_safe_crop = config['random_sized_bbox_safe_crop'] if ('random_sized_bbox_safe_crop' in config) else {'height':1024, 'width':1024, 'p': 0.0}

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
            self.channel_shuffle = config['channel_shuffle'] if ('channel_shuffle' in config) else {'p': 0.0} 
            self.random_gamma = config['random_gamma'] if ('random_gamma' in config) else {'p': 0.0}
            self.hsv = config['hsv'] if ('hsv' in config) else {'p': 0.0}
            self.rgb_shift = config['rgb_shift'] if ('rgb_shift' in config) else {'p': 0.0}
            self.random_brightness = config['random_brightness'] if ('random_brightness' in config) else {'p': 0.0}
            self.random_contrast = config['random_contrast'] if ('random_contrast' in config) else {'p': 0.0}
            
            # noise系
            self.gauss_noise = config['gauss_noise'] if ('gauss_noise' in config) else {'p': 0.0}
            self.cutout = config['cutout'] if ('cutout' in config) else {'p': 0.0}

        else:
            self.mosaic = {'p': 0.0}
            self.horizontal_flip = {'p': 0.0}
            self.vertical_flip = {'p': 0.0} 
            self.random_rotate_90 = {'p': 0.0}
            self.random_sized_bbox_safe_crop = {'height':1024, 'width':1024, 'p': 0.0}
            self.blur = {'p': 0.0}
            self.motion_blur = {'p': 0.0}
            self.median_blur = {'p': 0.0}
            self.gaussian_blur = {'p': 0.0}
            self.optical_distorion = {'p': 0.0}
            self.grid_distorion = {'p': 0.0}
            self.elastic_distorion = {'p': 0.0}
            self.clahe = {'p': 0.0}
            self.channel_shuffle = {'p': 0.0}
            self.random_gamma = {'p': 0.0}
            self.hsv = {'p': 0.0}
            self.rgb_shift = {'p': 0.0}
            self.random_brightness = {'p': 0.0}
            self.random_contrast = {'p': 0.0}
            self.gauss_noise = {'p': 0.0}
            self.cutout = {'p': 0.0}


    def __call__(self, example, dataset):

        image, target, image_id = example

        # mosaic
        if np.random.rand() < self.mosaic['p']:

            # get other datas from dataset (for mosaic)
            mosaic_image_sources = [dataset.get_example(i) for i in np.random.choice(np.arange(len(dataset.image_ids)), 3, replace=False)]

            image_list = [image]
            target_list = [target]
            image_id_list = [image_id]
            for source in mosaic_image_sources:
                image_list.append(source[0])
                target_list.append(source[1])
                image_id_list.append(source[2])
            
            # apply mosaic
            image, boxes, image_id = moisac(image_list, target_list, image_id_list)

            # recalculate area and label and iscrowed
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # only one class (background or wheet)        
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # suppose all instances are not crowd

            target['boxes'] = boxes
            target['labels'] = labels
            # target['image_id'] = torch.tensor([image_id])  # use base image image_id (concat image_id raise length is too long error)
            target['area'] = area
            target['iscrowd'] = iscrowd


        # for albumentation transforms
        sample = {
            'image': (image*255).astype(np.uint8),
            'bboxes': target['boxes'],
            'labels': target['labels']
        }

        albumentation_transforms = A.Compose([
            A.HorizontalFlip(**self.horizontal_flip),
            A.VerticalFlip(**self.vertical_flip),
            A.RandomRotate90(**self.random_rotate_90),
            A.RandomSizedBBoxSafeCrop(**self.random_sized_bbox_safe_crop),
            A.Blur(**self.blur),
            A.MotionBlur(**self.motion_blur),
            A.MedianBlur(**self.median_blur),
            A.GaussianBlur(**self.gaussian_blur),
            A.OpticalDistortion(**self.optical_distorion),
            A.GridDistortion(**self.grid_distorion),
            A.ElasticTransform(**self.elastic_distorion),
            A.CLAHE(**self.clahe),
            A.ChannelShuffle(**self.channel_shuffle),
            A.RandomGamma(**self.random_gamma),
            A.HueSaturationValue(**self.hsv),
            A.RGBShift(**self.rgb_shift),
            A.RandomBrightness(**self.random_brightness),
            A.RandomContrast(**self.random_contrast),
            A.GaussNoise(**self.gauss_noise),
            A.Cutout(**self.cutout),
            A.Resize(height=self.img_size, width=self.img_size, p=1.0 if self.model_name=='efficient_det' else 0.0), # GPU will be OOM without this
            ToTensorV2(p=1.0)  # convert numpy image to tensor
            ], 
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

        sample = albumentation_transforms(**sample)
        image = (sample['image'].type(torch.float32))/255
        target['boxes'] = torch.stack(tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)
        return image, target, image_id