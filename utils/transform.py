import argparse
import os
from pathlib import Path
import random
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
from collections import namedtuple


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
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox


from utils.functions import random_box, calc_box_overlap


def tile4(image_list, target_list, image_id_list):

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

    

def mosaic(image, target, image_id, dataset, p):
    
    if p < np.random.rand():
        return image, target

    additional_image = 3

    # get other datas from dataset (for mosaic)
    mosaic_image_sources = [dataset.get_example(i) for i in np.random.choice(np.arange(len(dataset.image_ids)), additional_image, replace=False)]

    source_image_list = [image for _ in range(4 - additional_image)] 
    source_target_list = [target for _ in range(4 - additional_image)] 
    source_image_id_list = [image_id for _ in range(4 - additional_image)] 
    for source in mosaic_image_sources:
        source_image_list.append(source[0])
        source_target_list.append(source[1])
        source_image_id_list.append(source[2])
    
    # shulle source
    image_list = []
    target_list = []
    image_id_list = []
    for i in np.random.permutation(np.arange(4)):
        image_list.append(source_image_list[i])
        target_list.append(source_target_list[i])
        image_id_list.append(source_image_id_list[i])

    # apply mosaic
    image, boxes, image_id = tile4(image_list, target_list, image_id_list)

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

    return image, target



def cutmix(image, target, image_id, dataset, p, mix=False, alpha=0.5, alpha2=2.0, keep_threshold=0.5):

    if p < np.random.rand():
        return image, target

    # beta分布 beta(alpha, alpha)からサンプリングしてboxを作る

    org_image = copy.deepcopy(image)
    org_target = copy.deepcopy(target)

    l = np.random.beta(alpha, alpha)
    l2 = np.random.beta(alpha2, alpha2)
    bbx1, bby1, bbx2, bby2 = random_box(image.shape[0], image.shape[1], l)
    cut_box = np.array([bbx1, bby1, bbx2, bby2])
    
    source = dataset.get_example(np.random.choice(np.arange(len(dataset.image_ids))))
    
    if mix:
        image[bby1:bby2, bbx1:bbx2] = image[bby1:bby2, bbx1:bbx2] * l2 + source[0][bby1:bby2, bbx1:bbx2] * (1 - l2)
        src_boxes = source[1]['boxes']
        src_keep_idx = np.where(calc_box_overlap(src_boxes, cut_box) >(1.0 - keep_threshold))[0]
        boxes = np.concatenate([target['boxes'], src_boxes[src_keep_idx, :]], axis=0)
    else:
        image[bby1:bby2, bbx1:bbx2] = source[0][bby1:bby2, bbx1:bbx2]

        src_boxes = source[1]['boxes']
        org_keep_idx = np.where(calc_box_overlap(target['boxes'], cut_box) < keep_threshold)[0]
        src_keep_idx = np.where(calc_box_overlap(src_boxes, cut_box) >(1.0 - keep_threshold))[0]

        boxes = np.concatenate([target['boxes'][org_keep_idx, :], src_boxes[src_keep_idx, :]], axis=0)
    
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    area = torch.as_tensor(area, dtype=torch.float32)
    labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # only one class (background or wheet)        
    iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # suppose all instances are not crowd

    target['boxes'] = boxes
    target['labels'] = labels
    target['area'] = area
    target['iscrowd'] = iscrowd
    
    if boxes.shape[0] == 0:
        return org_image, org_target
    else:
        return image, target


def mixup(image, target, image_id, dataset, p, alpha=2.0):

    if p < np.random.rand():
        return image, target

    l = np.random.beta(alpha, alpha)
    source = dataset.get_example(np.random.choice(np.arange(len(dataset.image_ids))))
    
    image = image * l + source[0] * (1 - l)
    src_boxes = source[1]['boxes']
    boxes = np.concatenate([target['boxes'], src_boxes], axis=0)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    area = torch.as_tensor(area, dtype=torch.float32)
    labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # only one class (background or wheet)        
    iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # suppose all instances are not crowd

    target['boxes'] = boxes
    target['labels'] = labels
    target['area'] = area
    target['iscrowd'] = iscrowd

    return image, target


class CustomCutout(DualTransform):
    """
    Custom Cutout augmentation with handling of bounding boxes 
    Note: (only supports square cutout regions)
    
    Author: Kaushal28
    Reference: https://arxiv.org/pdf/1708.04552.pdf
    """
    
    def __init__(
        self,
        fill_value=0,
        bbox_removal_threshold=0.50,
        min_cutout_size=60,
        max_cutout_size=70,
        always_apply=False,
        p=0.5
    ):
        """
        Class construstor
        
        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
        :param min_cutout_size: minimum size of cutout (192 x 192)
        :param max_cutout_size: maximum size of cutout (512 x 512)
        """
        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size
        
    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            np.random.randint(0, img_width - cutout_size + 1),
            np.random.randint(0, img_height - cutout_size + 1)
        )
    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position
        
    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image
        
        :param image: The image to be augmented
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, _ = image.shape
        cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)
        
        # Set to instance variables to use this later
        self.image = image
        self.cutout_pos = cutout_pos
        self.cutout_size = cutout_size
        
        image[cutout_pos.y:cutout_pos.y+cutout_size, cutout_pos.x:cutout_size+cutout_pos.x, :] = cutout_arr
        return image
    
    def apply_to_bbox(self, bbox, **params):
        """
        Removes the bounding boxes which are covered by the applied cutout       
        :param bbox: A single bounding box coordinates in pascal_voc format
        :returns transformed bbox's coordinates
        """

        # Denormalize the bbox coordinates
        bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
        x_min, y_min, x_max, y_max = tuple(map(int, bbox))

        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
        overlapping_size = np.sum(
            (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
        )

        # Remove the bbox if it has more than some threshold of content is inside the cutout patch
        if overlapping_size / bbox_size > self.bbox_removal_threshold:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        return normalize_bbox(bbox, self.img_height, self.img_width)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')


class Transform:
    """
    https://github.com/katsura-jp/tour-of-albumentations 参考
    """

    def __init__(self, all_config, is_train):
        self.model_name = all_config['model']['name']
        self.img_size = all_config['model']['config']['image_size'] if self.model_name=='efficient_det' else 1024
        config = all_config['train']['augment']
        if is_train or config['test_time_augment']:

            # mix and cut
            self.mosaic = config['mosaic'] if ('mosaic' in config) else {'p': 0.0}
            self.cutmix = config['cutmix'] if ('cutmix' in config) else {'p': 0.0}
            self.mixup = config['mixup'] if ('mixup' in config) else {'p': 0.0}
            self.cutout = config['cutout'] if ('cutout' in config) else {'p': 0.0}
            self.custom_cutout = config['custom_cutout'] if ('custom_cutout' in config) else {'p': 0.0}


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
            

        else:
            self.mosaic = {'p': 0.0}
            self.cutmix = {'p': 0.0}
            self.cutout = {'p': 0.0}
            self.custom_cutout = {'p': 0.0}
            self.mixup = {'p': 0.0}
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
            

    def __call__(self, example, dataset):

        image, target, image_id = example
        
        # mosaic, cutmix, mixup            
        image, target = mosaic(image, target, image_id, dataset, **self.mosaic)
        image, target = cutmix(image, target, image_id, dataset, **self.cutmix)
        image, target = mixup(image, target, image_id, dataset, **self.mixup)

        # for albumentation transforms
        sample = {
            'image': (image*255).astype(np.uint8),
            'bboxes': target['boxes'],
            'labels': target['labels']
        }

        albumentation_transforms = A.Compose([
            A.Cutout(**self.cutout),
            CustomCutout(**self.custom_cutout),
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
            A.Resize(height=self.img_size, width=self.img_size, p=1.0 if self.model_name=='efficient_det' else 0.0), # GPU will be OOM without this
            ToTensorV2(p=1.0)  # convert numpy image to tensor
            ], 
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

        sample = albumentation_transforms(**sample)
        image = (sample['image'].type(torch.float32))/255
        if sample['bboxes'] == []:
            # If empty after transform, fill out with these values.
            target['boxes'] = torch.tensor([[0,0,0,0]], dtype=torch.float32)
            target['labels'] = torch.tensor([1], dtype=torch.int64)
            target['area'] = torch.tensor([0], dtype=torch.float32)
            target['iscrowd'] = torch.tensor([0], dtype=torch.int64)
        else:
            target['boxes'] = torch.stack(tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id