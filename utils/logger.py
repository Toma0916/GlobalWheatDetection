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


class LossAverager:
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



class MetricAverager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class ImageStorage():

    def __init__(self):
        self.image_ids = None
        self.images = None
        self.target_boxes = None
        self.predict_boxes = None
        self.predict_scores = None

    def send(self, image_ids, images, target_boxes, predict_boxes, predict_scores):
        self.image_ids = image_ids
        self.images = images
        self.target_boxes = target_boxes
        self.predict_boxes = predict_boxes
        self.predict_scores = predict_scores

    @property
    def painted_images(self):

        id_image_dict = {}
        for i in range(len(self.image_ids)):
            image = self.images[i]
            for j in range(self.target_boxes[i].shape[0]):
                box = self.target_boxes[i][j]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220/255, 0, 0), 3)
            for j in range(self.predict_boxes[i].shape[0]):
                box = self.predict_boxes[i][j]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 220/255, 0), 3)
                cv2.putText(image, '%f' % self.predict_scores[i][j], (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 220/255, 0), 2, cv2.LINE_AA)
            id_image_dict[self.image_ids[i]] = image
        return id_image_dict    

    def reset(self):
        self.image_ids = None
        self.images = None
        self.target_boxes = None
        self.predict_boxes = None
        self.predict_scores = None



class TensorBoardLogger:

    def __init__(self, model, optimizer, output_dir, trained_epoch, model_save_interval):

        self.model = model 
        self.optimizer = optimizer
        self.trained_epoch = trained_epoch
        self.trained_epoch_this_run = 0
        self.model_save_interval = model_save_interval
        self.experiment_name = output_dir.name  # 保存するディレクトリ名を一致させる
        self.save_dir = output_dir

        self.train_loss_epoch_history = LossAverager()
        self.valid_loss_epoch_history = LossAverager()
        self.valid_metric_epoch_history = MetricAverager()
        self.valid_predict_image_epoch_history = ImageStorage()

        self.writer = SummaryWriter(log_dir=str(self.save_dir))

        self.mode = 'train'


    def __del__(self):
        pass

    def start_train_epoch(self):
        self.mode = 'train'
        self.train_loss_epoch_history.reset()

        # save leaering late for each epoch
        learning_rate = get_lr(self.optimizer)
        self.writer.add_scalar('train/lr', learning_rate, self.trained_epoch + 1)


    def end_train_epoch(self):
        for key, value in self.train_loss_epoch_history.value.items():
            self.writer.add_scalar('train/%s' % key, value, self.trained_epoch + 1)
        self.trained_epoch += 1
        self.trained_epoch_this_run += 1
        
        # save model snapshot
        if (self.trained_epoch_this_run - 1) % self.model_save_interval == 0:
            filepath = str(self.save_dir/('model_epoch_%s.pt' % str(self.trained_epoch).zfill(3)))
            torch.save(self.model.state_dict(), filepath)


    def start_valid_epoch(self):
        self.mode = 'valid'
        self.valid_loss_epoch_history.reset()
        self.valid_metric_epoch_history.reset()


    def end_valid_epoch(self):
        for key, value in self.valid_loss_epoch_history.value.items():
            self.writer.add_scalar('valid/%s' % key, value, self.trained_epoch)        
        self.writer.add_scalar('valid/score', self.valid_metric_epoch_history.value, self.trained_epoch)

        images = self.valid_predict_image_epoch_history.painted_images
        for key, value in images.items():
            self.writer.add_image('valid/%d/%s' % (self.trained_epoch ,key), value, global_step=self.trained_epoch, dataformats='HWC')

    def send_loss(self, loss_dict):
        if self.mode == 'train':
            self.train_loss_epoch_history.send(loss_dict)
        elif self.mode == 'valid':
            self.valid_loss_epoch_history.send(loss_dict)


    def send_score(self, score):
        """
        score: float
        """
        self.valid_metric_epoch_history.send(score)
    

    def send_images(self, images, image_ids, targets, outputs):
        images = [np.transpose(image.cpu().detach().numpy(), (1, 2, 0)) for image in images]
        target_boxes = [target['boxes'].detach().cpu().numpy().astype(np.int) for target in targets]
        predict_boxes = [output['boxes'].detach().cpu().numpy() for output in outputs]
        predict_scores = [output['scores'].detach().cpu().numpy() for output in outputs]
        self.valid_predict_image_epoch_history.send(image_ids, images, target_boxes, predict_boxes, predict_scores)
