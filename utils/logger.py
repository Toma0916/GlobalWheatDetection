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

# --- mlflow ---
import mlflow

from utils.functions import get_lr, params_to_mlflow_format, randomname


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
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.70, 0.75]
        self.current_total = {
            'original': np.zeros(len(self.iou_thresholds)),
            'processed': np.zeros(len(self.iou_thresholds))
        }
        self.iterations = 0.0

    def send(self, values, type):
        self.current_total[type] += values
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return {
                'original': 0,
                'processed': 0
            }
        else:
            return {
                'original': 1.0 * self.current_total['original'] / self.iterations,
                'processed': 1.0 * self.current_total['processed'] / self.iterations
            }
            

    def reset(self):
        self.current_total = {
            'original': np.zeros(len(self.iou_thresholds)),
            'processed': np.zeros(len(self.iou_thresholds))
        }
        self.iterations = 0.0


class ImageStorage():

    def __init__(self):
        self.image_ids = None
        self.images = None
        self.target_boxes = None
        self.original_predict_boxes = None
        self.original_predict_scores = None
        self.processed_predict_boxes = None
        self.processed_predict_scores = None

    def send(self, image_ids, images, target_boxes, original_predict_boxes, original_predict_scores, processed_predict_boxes, processed_predict_scores):
        self.image_ids = image_ids
        self.images = images
        self.target_boxes = target_boxes
        self.original_predict_boxes = original_predict_boxes
        self.original_predict_scores = original_predict_scores
        self.processed_predict_boxes = processed_predict_boxes
        self.processed_predict_scores = processed_predict_scores

    @property
    def painted_images(self):

        id_image_dict = {}
        for i in range(len(self.image_ids)):
            image = self.images[i]
            image = cv2.UMat(image).get()
            if self.target_boxes is not None:            
                for j in range(self.target_boxes[i].shape[0]):
                    box = self.target_boxes[i][j]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220/255, 0, 0), 3)

            if (self.original_predict_boxes is not None) and (self.original_predict_scores is not None):            
                for j in range(self.original_predict_scores[i].shape[0]):
                    box = self.original_predict_boxes[i][j]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220/255, 220/255, 0), 1)
                    cv2.putText(image, '%f' % self.original_predict_scores[i][j], (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 2.0, (220/255, 220/255, 0), 1, cv2.LINE_AA)
        
            if (self.processed_predict_boxes is not None) and (self.processed_predict_scores is not None):            
                for j in range(self.processed_predict_scores[i].shape[0]):
                    box = self.processed_predict_boxes[i][j]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 220/255, 0), 3)
                    cv2.putText(image, '%f' % self.processed_predict_scores[i][j], (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 220/255, 0), 2, cv2.LINE_AA)

            id_image_dict[self.image_ids[i]] = image
        return id_image_dict    

    def reset(self):
        self.image_ids = None
        self.images = None
        self.target_boxes = None
        self.original_predict_boxes = None
        self.original_predict_scores = None
        self.processed_predict_boxes = None
        self.processed_predict_scores = None



class Logger:

    def __init__(self, model, optimizer, output_dir, trained_epoch, config):

        self.model = model 
        self.optimizer = optimizer
        self.trained_epoch = trained_epoch
        self.trained_epoch_this_run = 0
        self.model_save_interval = config['general']['model_save_interval']
        self.valid_image_save_interval = config['general']['valid_image_save_interval']
        self.experiment_name = config['general']['experiment_name']
        self.save_dir = output_dir

        self.train_loss_epoch_history = LossAverager()
        self.valid_loss_epoch_history = LossAverager()
        self.valid_metric_epoch_history = MetricAverager()
        self.image_epoch_history = ImageStorage()

        self.writer = SummaryWriter(log_dir=str(self.save_dir))
        self.mode = 'train'

        self.initialize_mlflow(config)


    def initialize_mlflow(self, config):
        # mlflow
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name='%s_%s' % (self.experiment_name, randomname(4)))
        mlflow.log_params(params_to_mlflow_format(config))
        mlflow.log_artifact(str(self.save_dir/"config.json"))

    def finish_training(self):
        self.writer.close()
        mlflow.end_run()


    def start_train_epoch(self):
        self.mode = 'train'
        self.train_loss_epoch_history.reset()
        self.image_epoch_history .reset()

        # save leaering late for each epoch
        learning_rate = get_lr(self.optimizer)
        self.writer.add_scalar('train/lr', learning_rate, self.trained_epoch + 1)

    def end_train_epoch(self):
        for key, value in self.train_loss_epoch_history.value.items():
            self.writer.add_scalar('train/%s' % key, value, self.trained_epoch + 1)
            mlflow.log_metrics({'tr_%s' % (key.replace('loss', 'ls')) : value})
        self.trained_epoch += 1
        self.trained_epoch_this_run += 1

        if self.trained_epoch == 1:  # 最初のepochだけtrain画像を保存する
            images = self.image_epoch_history.painted_images
            for key, value in images.items():
                self.writer.add_image('train/sample/%s' % (key), value, global_step=self.trained_epoch, dataformats='HWC')
        
        # save model snapshot
        if (self.trained_epoch_this_run - 1) % self.model_save_interval == 0:
            filepath = str(self.save_dir/('model_epoch_%s.pt' % str(self.trained_epoch).zfill(3)))
            torch.save(self.model.state_dict(), filepath)
        
    def start_valid_epoch(self):
        self.mode = 'valid'
        self.valid_loss_epoch_history.reset()
        self.valid_metric_epoch_history.reset()
        self.image_epoch_history.reset()


    def end_valid_epoch(self):

        for key, value in self.valid_loss_epoch_history.value.items():
            self.writer.add_scalar('valid/%s' % key, value, self.trained_epoch)    
            mlflow.log_metrics({'val_%s' % (key.replace('loss', 'ls')) : value})

        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.70, 0.75]
        metrics_scores = self.valid_metric_epoch_history.value
        for key, values in metrics_scores.items():
            scores = []
            for i, value in enumerate(values):
                scores.append(value)
                self.writer.add_scalar('valid/%s_%.2f' % (key, iou_thresholds[i]), value, self.trained_epoch)
                mlflow.log_metrics({'val_%s_%.2f' % ('org' if key == 'original' else 'prc' , iou_thresholds[i]): value})
            self.writer.add_scalar('valid/%s_average' % (key), sum(scores)/len(scores), self.trained_epoch)
            mlflow.log_metrics({'val_%s_avg' % ('org' if key == 'original' else 'prc'):  sum(scores)/len(scores)})

        # save image
        if ((self.trained_epoch_this_run - 1)) % self.valid_image_save_interval == 0:
            images = self.image_epoch_history.painted_images
            for key, value in images.items():
                self.writer.add_image('valid/%d/%s' % (self.trained_epoch ,key), value, global_step=self.trained_epoch, dataformats='HWC')
        

    def send_loss(self, loss_dict):
        if self.mode == 'train':
            self.train_loss_epoch_history.send(loss_dict)
        elif self.mode == 'valid':
            self.valid_loss_epoch_history.send(loss_dict)


    def send_score(self, scores, score_type):
        """
        score: float
        """
        assert score_type in ['original', 'processed']
        self.valid_metric_epoch_history.send(scores, score_type)
    

    def send_images(self, images, image_ids, target_boxes=None, original_outputs=None, processed_outputs=None):
        images = [np.transpose(image.cpu().detach().numpy(), (1, 2, 0)) for image in images]
        original_predict_boxes = [output['boxes'] for output in original_outputs] if (original_outputs is not None) else None
        original_predict_scores = [output['scores'] for output in original_outputs] if (original_outputs is not None) else None
        processed_predict_boxes = [output['boxes'] for output in processed_outputs] if (processed_outputs is not None) else None
        processed_predict_scores = [output['scores'] for output in processed_outputs] if (processed_outputs is not None) else None

        self.image_epoch_history.send(image_ids, images, target_boxes, original_predict_boxes, original_predict_scores, processed_predict_boxes, processed_predict_scores)




