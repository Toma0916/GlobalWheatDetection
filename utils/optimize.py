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
import shutil
import tqdm
import copy
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

from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from skimage.transform import AffineTransform, warp
import sklearn.metrics

from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret
from skopt.space import Categorical, Integer, Real

from utils.postprocess import postprocessing
from utils.metric import calculate_score_for_each
from utils.functions import get_config


class PostProcessOptimizer():

    def __init__(self, train_predicts, valid_predicts):
        self.train_predicts = copy.deepcopy(train_predicts)
        self.valid_predicts = copy.deepcopy(valid_predicts)
        self.optimization_result = {}

    def optimize(self, name, space, n_calls=10):

        @use_named_args(space)
        def score(**params):
            scores = []
            for image_id in self.train_predicts.keys():
                processed = postprocessing([self.train_predicts[image_id]['original']], get_config(name, **params))
                scores.append(calculate_score_for_each(processed, [self.train_predicts[image_id]['target']]))
            final_score = np.mean(np.array(scores))            
            return -final_score

        return gp_minimize(func=score, dimensions=space, n_calls=n_calls)
    

    def optimize_nms(self, n_calls=10):
        print('optimizing nms...')
        space = [
            Real(0, 1, name='threshold'),
            Real(0, 1, name='min_confidence')
        ]
        opt_result = self.optimize(name='nms', space=space, n_calls=n_calls)
        self.optimization_result['nms'] = dict(opt_result)

        best_itr = np.argmin(opt_result.func_vals)
        best_params = opt_result.x_iters[best_itr]
        return best_params
    
    def optimize_soft_nms(self, n_calls=10):
        print('optimizing soft nms...')
        space = [
            Real(0, 1, name='sigma'),
            Real(0, 1, name='min_confidence')
        ]
        opt_result = self.optimize(name='soft_nms', space=space, n_calls=n_calls)
        self.optimization_result['soft_nms'] = dict(opt_result)

        best_itr = np.argmin(opt_result.func_vals)
        best_params = opt_result.x_iters[best_itr]
        return best_params
    
    
    def optimize_wbf(self, n_calls=10):
        print('optimizing wbf...')
        space = [
            Real(0, 1, name='threshold'),
            Real(0, 1, name='min_confidence')
        ]
        opt_result = self.optimize(name='wbf', space=space, n_calls=n_calls)
        self.optimization_result['wbf'] = dict(opt_result)
        
        best_itr = np.argmin(opt_result.func_vals)
        best_params = opt_result.x_iters[best_itr]
        return best_params    

