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
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from skimage.transform import AffineTransform, warp
import sklearn.metrics

# --- albumentations ---
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import DualTransform

# --- EfficientDet ---
from utils.effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from utils.effdet.efficientdet import HeadNet

from utils.functions import xywh2xyxy

def efficientdet_model(image_size, pretrained_path=None, class_num=1):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=True)
    if not pretrained_path is None:
        checkpoint = torch.load(pretrained_path)
        net.load_state_dict(checkpoint)
    config.num_classes = class_num
    config.image_size = image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

def fasterrcnn_model(backbone, class_num=2, pool_layers_num=4, pooled_size=7, pretrained=True):
    """
    pool_layers_num: MultiScaleRoIAlignで使う層の数. 'resnet50_coco'では無視される. 安全そうな範囲で1~4で指定
    pooled_size: RoIPool後のmap size. 'resnet50_coco'では無視される. 安全そうな5~9で指定
    """

    backbone_list = {
        'resnet18': True,
        'resnet34': True,
        'resnet50': True,
        'resnet50_coco': True,  # いままでのやつ、headまでCOCOでpretrained
        'resnet101': True,  # batch_size=4は乗る
        'resnet152': True,   # batch_size=4は乗る
        'resnext50_32x4d': True,
        # 'resnext101_32x8d': True,  # エラー起きる
        # 'wide_resnet50_2': True,  # エラー起きる
        # 'wide_resnet101_2': True  # エラー起きる
    }

    assert backbone in backbone_list.keys(), 'Backbone\'s name is not valid. Available backbones: %s' % str(list(backbone_list.keys()))
    if backbone == 'resnet50_coco':
        # 今まで使っていたmodel、headまでpretrainedでweightsを読み込んでおり構造は弄れない
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features	
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)

    else:
        # backboneだけpretrained
        assert pool_layers_num in [1, 2, 3, 4], 'pool_layers_num must be in [1, 2, 3, 4] You selected %d' % (pool_layers_num) 
        assert pooled_size in [5, 6, 7, 8, 9], 'pooled_size must be in [5, 6, 7, 8, 9] You selected %d' % (pooled_size) 

        # anchor_sizesはデフォルトから1スケール落とした。 default: ((32,), (64,), (128,), (256,), (512,))
        anchor_sizes = ((16), (32,), (64,), (128,), (256,))
        # anchor_ratiosは4:1の比を追加
        aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        # デフォルトでマルチスケールのRoIAlignになっている。headに近い4層から特徴を抽出しているはず
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[str(n) for n in range(pool_layers_num)], output_size=pooled_size, sampling_ratio=2)
        backbone = resnet_fpn_backbone(backbone, pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=class_num, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

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
        self.image_size = config['config']['image_size']  if 'image_size' in config['config'].keys() else 1024
        self.domain_loss_coefficient = 0.0 if ('domain_loss_coefficient' not in config['config']) else config['config']['domain_loss_coefficient']  # now, used in only faster rcnn

        # Used for efficientdet
        self.resize_transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1.0 if self.model_name=='efficient_det' else 0.0), # GPU will be OOM without this
            ToTensorV2(p=1.0)  # convert numpy image to tensor
            ], 
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )
        self.resize_back_transform = A.Compose([
            A.Resize(height=1024, width=1024), # GPU will be OOM without this
            ToTensorV2(p=1.0)  # convert numpy image to tensor
            ], 
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

        self.dl_calculator = DomainLossCalculator()


    def __call__(self, images, targets=None):
        images, targets = self._resize(images, targets)
        if self.model_name == 'faster_rcnn':
            images = list(image.float().to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            if self.is_train:
                loss, pooled_features = self.model(images, targets)
                loss['loss_domain'] = self.domain_loss_coefficient * self.dl_calculator(pooled_features, targets)
                return loss
            else:
                self.model.train()
                loss_dict, pooled_features = self.model(images, targets)
                loss_dict['loss_domain'] = self.domain_loss_coefficient * self.dl_calculator(pooled_features, targets)

                self.model.eval()
                preds = self.model(images, targets)
                images, preds = self._resize_back(images, preds)
                preds = [{k: v.cpu().detach() for k, v in pred.items()} for pred in preds]
                return preds, loss_dict

        elif self.model_name == 'efficient_det':
            # resize images and boxes into self.image_size
            images = torch.stack(images).to(self.device).float()
            boxes = [target['boxes'][:, [1, 0, 3, 2]].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]
            if self.is_train:
                loss, _, _ = self.model(images, boxes, labels)
                return {'loss': loss}
            else:
                outputs, (loss, _, _) = self.model(images, boxes, labels)
                preds = self._preds_from_effdet_output(outputs)
                
                images, preds = self._resize_back(images, preds)
                preds = [{k: v.cpu().detach() for k, v in pred.items()} for pred in preds]
                return preds, {'loss': loss}

    def to(self, device):
        self.model.to(device)
        self.dl_calculator.to(device)
        self.device = device
        return self

    def eval(self):
        self.model.eval()
        self.dl_calculator.eval()
        self.is_train = False

    def train(self):
        self.model.train()
        self.dl_calculator.train()
        self.is_train = True

    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))
        return self

    def _preds_from_effdet_output(self, outputs):
        preds = []
        for i in range(outputs.shape[0]):
            pred = ({
                'boxes': xywh2xyxy(outputs[i, :, :4]).clamp(min=0, max=self.image_size-1),
                'labels': outputs[i, :, 5],
                'scores': outputs[i, :, 4]
            })
            pred['labels'] = pred['labels'][pred['boxes'].sum(axis=1)>0]
            pred['scores'] = pred['scores'][pred['boxes'].sum(axis=1)>0]
            pred['boxes'] = pred['boxes'][pred['boxes'].sum(axis=1)>0]
            preds.append(pred)
        return preds

    def _resize(self, images, targets):
        if images[0].shape[1:] == (self.image_size, self.image_size):
            return images, targets

        samples = [{
            'image': image.permute(1, 2, 0).cpu().numpy(),
            'bboxes': target['boxes'],
            'labels': target['labels']
        } for image, target in zip(images, targets)]
        samples = [self.resize_transform(**sample) for sample in samples]
        targets_resized = list(targets)
        for i, (target, sample) in enumerate(zip(targets, samples)):
            if len(sample['bboxes'])!=0:
                target['boxes'] = torch.stack(tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)
            else:
                target['boxes'] = torch.tensor([])
            targets_resized[i] = target
        images_resized = [sample['image'] for sample in samples]
        return images_resized, targets_resized

    def _resize_back(self, images, outputs):
        if images[0].shape[1:] == (self.image_size, self.image_size):
            return images, outputs

        samples = [{
            'image': image.permute(1, 2, 0).cpu().numpy(),
            'bboxes': output['boxes'],
            'labels': output['labels']
        } for image, output in zip(images, outputs)]
        samples = [self.resize_back_transform (**sample) for sample in samples]
        outputs_resized = outputs
        for i, (output, sample) in enumerate(zip(outputs, samples)):
            if len(sample['bboxes'])!=0:
                output['boxes'] = torch.stack(tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)
            else:
                output['boxes'] = torch.tensor([])
            outputs_resized[i] = output
        images_resized = [sample['image'] for sample in samples]
        return images_resized, outputs_resized



class DomainLossCalculator(nn.Module):
    """
    Implementaion of Idea of DANN
    predict source domain of image by using top feature of backbone
    """

    sources_label_map = {
            'usask_1': 0,
            'arvalis_1': 1,
            'inrae_1': 2,
            'ethz_1': 3,
            'arvalis_3': 4,
            'rres_1': 5,
            'arvalis_2': 6
    }
    train_domain_num = len(sources_label_map.keys())

    def __init__(self):
        
        super(DomainLossCalculator, self).__init__()
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256, 128))  # [warn]: `256` maybe give an error when using non default model
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(128))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(128, self.train_domain_num))


    def forward(self, features, targets):
        domain_output = self.domain_classifier(self.spacial_average_pooling_2d(features))    
        source_labels = torch.cat([target['source'] for target in targets])
        loss = nn.CrossEntropyLoss()(domain_output, source_labels) * (-1)  # reverse value because we intends model not to learn domain classification
        return loss
    

    def spacial_average_pooling_2d(self, features):
        """
        simple average pooling 
        """
        features = torch.mean(features, axis=2)
        features = torch.mean(features, axis=2)
        return features







