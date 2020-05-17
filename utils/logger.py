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

from collections import defaultdict

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

from utils.functions import get_lr


class Averager:
    def __init__(self):
        self.current_total =  defaultdict(float)
        self.iterations = 0.0

    def send(self, dictionary):
        for key, value in dictionary.items():
            self.current_total[key] += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return None
        else:
            losses = 0.0
            for value in self.current_total.values():
                losses += value 
            self.current_total['loss'] = losses

            loss_values_dict = dict()
            for key in self.current_total.keys():
                loss_values_dict[key] = self.current_total[key] / self.iterations
            return loss_values_dict

    def reset(self):
        for key in self.current_total.keys():
            self.current_total[key] = 0.0
        self.iterations = 0.0



class TensorBoardLogger:

    def __init__(self, config):
        
        self.config = config

        self.trained_epoch = config['trained_epoch']
        self.trained_iter = config['trained_iter']
        
        self.experiment_name = Path(config['output_dir']).name  # 保存するディレクトリ名を一致させる
        self.save_dir = Path(config['output_dir'])

        assert os.path.exists(str(self.save_dir))  is False
        os.makedirs(str(self.save_dir), exist_ok=False)

        self.train_loss_epoch_history = Averager()
        self.valid_loss_epoch_history = Averager()
        self.valid_score_epoch_history = Averager()

        self.writer = SummaryWriter(log_dir=str(self.save_dir))
        self.log_configs(config)

        self.mode = 'train'


    def __del__(self):
        pass

    def start_train_epoch(self, optimizer):
        self.mode = 'train'
        self.train_loss_epoch_history.reset()

        # save leaering late for each epoch
        learning_rate = get_lr(optimizer)
        self.writer.add_scalar('train/lr', learning_rate, self.trained_epoch + 1)



    def end_train_epoch(self):
        for key, value in self.train_loss_epoch_history.value.items():
            self.writer.add_scalar('train/%s' % key, value, self.trained_epoch + 1)
        self.trained_epoch += 1
    

    def start_valid_epoch(self):
        self.mode = 'valid'
        self.valid_loss_epoch_history.reset()
    

    def end_valid_epoch(self):
        for key, value in self.valid_loss_epoch_history.value.items():
            self.writer.add_scalar('valid/%s' % key, value, self.trained_epoch)
        


    def send(self, loss_dict):
        if self.mode == 'train':
            self.train_loss_epoch_history.send(loss_dict)
        elif self.mode == 'valid':
            self.valid_loss_epoch_history.send(loss_dict)


    def send_score(self, score):
        """score: float
        """
        self.writer.add_scalar('valid/score', score, self.trained_epoch + 1)


    def log_configs(self, config):

        # save config
        with open(self.save_dir/'train_config.json', 'w') as f:
            json.dump(config, f)

        for key, value in config.items():
            if 'train-' in key:
                key = key.replace('train-', 'train_augument/')
            elif 'valid-' in key:
                key = key.replace('valid-', 'valid_augument/')
            else:
                key = 'general/%s' % key
            self.writer.add_text('config/%s' % key, str(value))
    
