import argparse
import os
from pathlib import Path
import random
import math
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

### Define custom lr schedulers. Might be better to make another file like 'utils/schedulers.py' for the classes below.
# from: https://github.com/lyakaap/pytorch-template/blob/master/src/lr_scheduler.py
from bisect import bisect_right

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


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """multi-step learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be main.tex list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """cosine annealing scheduler with warmup.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )

        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return self.get_lr_warmup()
        else:
            return self.get_lr_cos_annealing()

    def get_lr_warmup(self):
        if self.warmup_method == "constant":
            warmup_factor = self.warmup_factor
        elif self.warmup_method == "linear":
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor
            for base_lr in self.base_lrs
        ]

    def get_lr_cos_annealing(self):
        last_epoch = self.last_epoch - self.warmup_iters
        T_max = self.T_max - self.warmup_iters
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * last_epoch / T_max)) / 2
                for base_lr in self.base_lrs]



class PiecewiseCyclicalLinearLR(torch.optim.lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group using piecewise
    cyclical linear schedule.
    When last_epoch=-1, sets initial lr as lr.
    
    Args:    
        c: cycle length
        alpha1: lr upper bound of cycle
        alpha2: lr lower bound of cycle
    _Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs
    https://arxiv.org/pdf/1802.10026
    _Exploring loss function topology with cyclical learning rates
    https://arxiv.org/abs/1702.04283
    """

    def __init__(self, optimizer, c, alpha1=1e-2, alpha2=5e-4, last_epoch=-1):

        self.c = c
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        super(PiecewiseCyclicalLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        lrs = []
        for _ in range(len(self.base_lrs)):
            ti = ((self.last_epoch - 1) % self.c + 1) / self.c
            if 0 <= ti <= 0.5:
                lr = (1 - 2 * ti) * self.alpha1 + 2 * ti * self.alpha2
            elif 0.5 < ti <= 1.0:
                lr = (2 - 2 * ti) * self.alpha2 + (2 * ti - 1) * self.alpha1
            else:
                raise ValueError('t(i) is out of range [0,1].')
            lrs.append(lr)

        return lrs


class PolyLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, power=0.9, max_epoch=4e4, last_epoch=-1):
        self.power = power
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * (1.0 - (self.last_epoch / self.max_epoch)) ** self.power
            lrs.append(lr)

        return lrs


class CosineAnnealingWarmUpRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def step_scheduler(optimizer, step_size=10, gamma=0.1, last_epoch=-1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

def mulitstep_scheduler(optimizer, milestones=[10, 20, 40], gamma=0.5):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones, gamma)

def exponential_scheduler(optimizer, gamma=0.95):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

def cosine_annealing_scheduler(optimizer, T_max=20, eta_min=0.001):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max, eta_min)

def warmup_multistep(optimizer, milestones=[10, 30, 50], gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, warmup_method="linear", last_epoch=-1):
    return WarmupMultiStepLR(optimizer, milestones, gamma, warmup_factor, warmup_iters, warmup_method, last_epoch)

def warmup_cosine_annealing_scheduler(optimizer, T_max=100, eta_min=1e-6, warmup_factor=1.0 / 3, warmup_iters=10, warmup_method="linear", last_epoch=-1):
    return WarmupCosineAnnealingLR(optimizer, T_max, eta_min, warmup_factor, warmup_iters, warmup_method, last_epoch)

def piecewise_cyclical_linear_scheduler(optimizer, c=10, alpha1=1e-2, alpha2=5e-4, last_epoch=-1):
    return PiecewiseCyclicalLinearLR(optimizer, c, alpha1, alpha2, last_epoch)

def poly_scheduler(optimizer, power=0.9, max_epoch=4e4, last_epoch=-1):
    return PolyLR(optimizer, power, max_epoch, last_epoch)

def warmup_cosine_annealing_restarts_scheduler(optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
    return CosineAnnealingWarmUpRestarts(optimizer, T_0, T_mult, eta_max, T_up, gamma, last_epoch)

def reduce_lr_on_plateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)

class CustomScheduler:
    def __init__(self, optimizer, config):
        scheduler_list = {
            'Step': step_scheduler,
            'MultiStep': mulitstep_scheduler,
            'Exponential': exponential_scheduler,
            'CosineAnnealing': cosine_annealing_scheduler,
            'WarmupMultiStep': warmup_multistep,
            'WarmupCosineAnnealing': warmup_cosine_annealing_scheduler,
            'PiecewiseCyclicalLinear': piecewise_cyclical_linear_scheduler,
            'Poly': poly_scheduler,
            'WarmupCosineAnnealingRestarts': warmup_cosine_annealing_restarts_scheduler,
            'ReduceLROnPlateau': reduce_lr_on_plateau
        }

        assert config['name'] in scheduler_list.keys(), 'Scheduler\'s name is not valid. Available schedulers: %s' % str(list(scheduler_list.keys()))
        self.scheduler_name = config['name']
        self.scheduler = scheduler_list[config['name']](optimizer, **config['config'])

    def step(self, metrics):
        if self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(metrics=metrics)
        else:
            self.scheduler.step()

            
def get_scheduler(config, optimizer):

    if config['name'] == '':
        return None

    scheduler = CustomScheduler(optimizer, config)
    return scheduler 
