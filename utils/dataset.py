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

from utils.functions import convert_dataframe, filter_bboxes_by_size, drop_bboxes_by_probability, vibrate_bboxes_with_ratio
from utils.transform import Transform


def collate_fn(batch):
    return tuple(zip(*batch))


class DatasetMixin(Dataset):

    def __init__(self, transform=None):
        self.transform = transform


    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)


    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError
      

    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)
        if self.transform:
            example = self.transform(example, self)  
        else:
            image, target, image_id = example
            image = np.transpose(image, (2, 0, 1))
            image = torch.as_tensor(image, dtype=torch.float32)
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
            example = (image, target, image_id)

        return example


    def get_example(self, i):
        """
        Returns the i-th example.
        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.
        Args:
            i (int): The index of the example.
        Returns:
            The i-th example.
        """
        raise NotImplementedError



class GWDDataset(DatasetMixin):

    def __init__(self, dataframe, image_dir, config=None, is_train=False, do_transform=True):
        
        self.config = config
        self.is_train = is_train
        
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.indices = np.arange(len(self.image_ids))
        self.image_size = 1024

        transform = Transform(self.config, self.is_train) if do_transform else None
        super(GWDDataset, self).__init__(transform=transform)

        dff = self.df[['image_id', 'source']].drop_duplicates()
        self.sources = dict(zip(dff.image_id, dff.source))

        self.sources_label_map = {
            'usask_1': 0,
            'arvalis_1': 1,
            'inrae_1': 2,
            'ethz_1': 3,
            'arvalis_3': 4,
            'rres_1': 5,
            'arvalis_2': 6
        }
     
    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)
    
    def get_example(self, i):

        im_h = self.image_size
        im_w = self.image_size

        image_id = self.image_ids[self.indices[i]]
        records = self.df[self.df['image_id'] == image_id]
        image = cv2.imread(str(self.image_dir/('%s.jpg' % image_id)), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        if self.is_train:
            boxes = drop_bboxes_by_probability(boxes, self.config['train']['random_annotation']['drop']['p'])
            boxes = vibrate_bboxes_with_ratio(boxes, self.config['train']['random_annotation']['vibration']['max_ratio'], (im_h, im_w))

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # only one class (background or wheet)        
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # suppose all instances are not crowd
        source = str(records[['source']].values[0][0])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([self.indices[i]])
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['source'] = torch.tensor([self.sources_label_map[source]])

        if not 'valid' in self.config.keys():
            self.config['valid'] = {'apply_bbox_filter': False}
        if self.is_train or self.config['valid']['apply_bbox_filter']:
            target = filter_bboxes_by_size(target, self.config['general']['bbox_filter'])
            
        # get another sample if number of bounding boxes is zero
        if len(target['boxes']) == 0:
            return self.get_example(np.random.randint(0, len(self)))

        return image, target, image_id


if __name__ == '__main__':

    src_dir = Path('./../src')
    train_image_dir = src_dir/'train'
    dataframe = pd.read_csv(str(src_dir/'train.csv'))
    dataframe = convert_dataframe(dataframe)

    dataset = GWDDataset(dataframe, train_image_dir)
    indices = torch.randperm(len(dataset)).tolist()
    
    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    