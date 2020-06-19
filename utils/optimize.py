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

import mlflow

from utils.postprocess import postprocessing
from utils.metric import calculate_score_for_each
from utils.functions import get_config, params_to_mlflow_format


class PostProcessOptimizer():

    def __init__(self, train_predicts, train_metrics, valid_predicts, valid_metrics):
        self.train_predicts = copy.deepcopy(train_predicts)
        self.train_metrics = copy.deepcopy(train_metrics)
        self.valid_predicts = copy.deepcopy(valid_predicts)
        self.valid_metrics = copy.deepcopy(valid_metrics)
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
        space = [
            Real(0, 1, name='threshold'),
            Real(0, 1, name='min_confidence')
        ]
        opt_result = self.optimize(name='nms', space=space, n_calls=n_calls)
        self.optimization_result['nms'] = dict(opt_result)

        best_itr = np.argmin(opt_result.func_vals)
        best_params = opt_result.x_iters[best_itr]
        best_params_dict = {
            'threshold': best_params[0],
            'min_confidence': best_params[1]
        }

        self.optimization_result['nms']['best_params_dict'] = best_params_dict
        return best_params_dict
    
    def optimize_soft_nms(self, n_calls=10):
        space = [
            Real(0, 1, name='sigma'),
            Real(0, 1, name='min_confidence')
        ]
        opt_result = self.optimize(name='soft_nms', space=space, n_calls=n_calls)
        self.optimization_result['soft_nms'] = dict(opt_result)

        best_itr = np.argmin(opt_result.func_vals)
        best_params = opt_result.x_iters[best_itr]
        best_params_dict = {
            'sigma': best_params[0],
            'min_confidence': best_params[1]
        }

        self.optimization_result['soft_nms']['best_params_dict'] = best_params_dict
        return best_params_dict
    
    
    def optimize_wbf(self, n_calls=10):
        space = [
            Real(0, 1, name='threshold'),
            Real(0, 1, name='min_confidence')
        ]
        opt_result = self.optimize(name='wbf', space=space, n_calls=n_calls)
        self.optimization_result['wbf'] = dict(opt_result)
        
        best_itr = np.argmin(opt_result.func_vals)
        best_params = opt_result.x_iters[best_itr]
        best_params_dict = {
            'threshold': best_params[0],
            'min_confidence': best_params[1]
        }

        self.optimization_result['wbf']['best_params_dict'] = best_params_dict
        return best_params_dict    

    
    def send(self, train_metrics, valid_metrics, method):
        self.optimization_result[method]['train_metrics'] = train_metrics
        self.optimization_result[method]['valid_metrics'] = valid_metrics


    def log(self, predict_config, general_config):

        # make log by mlflow                
        mlflow.set_tracking_uri('./mlruns')
        if not bool(mlflow.get_experiment_by_name('postprocessing')):
            mlflow.create_experiment('postprocessing', artifact_location=None)
        mlflow.set_experiment('postprocessing')
        mlflow.start_run()

        # log params
        params = {
            'debug': predict_config['debug'],
            'random_seed': general_config['general']['seed'],
            'train_valid_split': general_config['general']['train_valid_split']['name'],
            'pseudo_label': predict_config['pseudo_label']['apply'],
        }
        mlflow.log_params(params)

        # log used models as tags
        mlflow.set_tags({model_path: True for model_path in predict_config['model_paths']})

        # log metrics
        mlflow.log_metrics({'metrics_original_train': self.train_metrics, 'metrics_original_valid': self.valid_metrics})
        for method in ['nms', 'soft_nms', 'wbf']:
            if method not in self.optimization_result.keys():
                continue            
            method_metrics_dict = {}
            method_metrics_dict['metrics_%s_train' % method] = self.optimization_result[method]['train_metrics']
            method_metrics_dict['metrics_%s_valid' % method] = self.optimization_result[method]['valid_metrics']
            for key, value in self.optimization_result[method]['best_params_dict'].items():
                method_metrics_dict['p_%s_%s' % (method, key)] = value
            mlflow.log_metrics(method_metrics_dict)

        mlflow.end_run()
