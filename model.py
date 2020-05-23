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

# --- EfficientDet ---
from utils.effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from utils.effdet.efficientdet import HeadNet


def efficientdet_model(image_size, class_num=1):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    # checkpoint = torch.load('../input/efficientdet/efficientdet_d5-ef44aea8.pth')
    # net.load_state_dict(checkpoint)
    config.num_classes = class_num
    config.image_size = image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

def fasterrcnn_model(backbone, class_num=2, pretrained=True):

    backbone_list = {
        'fasterrcnn_resnet50_fpn': torchvision.models.detection.fasterrcnn_resnet50_fpn
    }

    assert backbone in backbone_list.keys(), 'Backbone\'s name is not valid. Available backbones: %s' % str(list(backbone_list.keys()))
    model = backbone_list[backbone](pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)
    return model 

class Model:
    def __init__(self, config):
        model_list = {
            'faster_rcnn': fasterrcnn_model,
            'efficient_det': efficientdet_model
        }
        assert config['name'] in model_list.keys(), 'Model\'s name is not valid. Available models: %s' % str(list(model_list.keys()))
        self.model_name = config['name']
        self.model = model_list[config['name']](**config['config'])
        self.is_train = True
        self.device = None
        self.image_size = (config['config']['image_size'], config['config']['image_size'])  if 'image_size' in config['config'].keys() else None

        # TODO: This is hardcoded 
        self.image_scale = self.image_size[0]/1024 

    def __call__(self, images, targets):
        if self.model_name == 'faster_rcnn':
            images = list(image.float().to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            if self.is_train:
                loss = self.model(images, targets)
                return loss
            else:
                self.model.train()
                loss_dict = self.model(images, targets)

                self.model.eval()
                preds = self.model(images, targets)
                return preds, loss_dict

        elif self.model_name == 'efficient_det':
            if self.is_train:
                new_targets = {}
                new_targets['bbox'] = [target['boxes'].to(self.device).float() for target in targets]
                new_targets['cls'] = [target['labels'].to(self.device).float() for target in targets]
                images = torch.stack(images).to(self.device).float()
                loss_dict = self.model(images, new_targets)
                return loss_dict
            else:
                new_targets = {}
                new_targets['bbox'] = [target['boxes'].to(self.device).float() for target in targets]
                new_targets['cls'] = [target['labels'].to(self.device).float() for target in targets]
                images = torch.stack(images).to(self.device).float()
                # TODO: Koko kirei ni suru
                new_targets['img_size'] = torch.full((len(images), 2), self.image_size[0]).to(self.device)
                new_targets['img_scale'] = (torch.ones(len(images),1) * self.image_scale).to(self.device)
                outputs = self.model(images, new_targets)
                preds = [{'boxes': res[:, :4],
                          'labels': res[:, 4],
                          'scores': res[:, 5]} for res in outputs['detections']]

                # preds['boxes'] = outputs['detections']
                loss_dict = {k: v for k, v in outputs.items() if 'loss' in k}
                return preds, loss_dict

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def eval(self):
        self.model.eval()
        self.is_train = False

    def train(self):
        self.model.train()
        self.is_train = True

    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()

def get_model(config):

    model_list = {
        'faster_rcnn': fasterrcnn_model,
        'efficient_det': efficientdet_model
    }

    assert config['name'] in model_list.keys(), 'Model\'s name is not valid. Available models: %s' % str(list(model_list.keys()))
    model = model_list[config['name']](**config['config'])
    return model