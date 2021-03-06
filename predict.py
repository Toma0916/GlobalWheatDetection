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
import itertools


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
from torch.utils.data import WeightedRandomSampler
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
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import DualTransform

# --- my modules ---
from model import Model
from optimizer import get_optimizer
from scheduler import get_scheduler

from utils.dataset import GWDDataset, collate_fn
from utils.transform import Transform
from utils.logger import Logger
from utils.functions import get_config, convert_dataframe, format_config_by_baseconfig, randomname
from utils.metric import calculate_score_for_each
from utils.postprocess import postprocessing
from utils.optimize import PostProcessOptimizer
from utils.sampler import get_sampler
from utils.train_valid_split import train_valid_split
from utils.pseudo_labeling import prepare_pseudo_labels, retrain_with_pseudo_label

warnings.simplefilter('ignore')  # 基本warningオフにしたい

def sanity_check(loaded_models):
    random_seed_list = []
    train_valid_split_list = []

    if len(loaded_models.keys()) == 1:
        return True

    for key in loaded_models.keys():
        random_seed_list.append(loaded_models[key]['config']['general']['seed'])
        train_valid_split_list.append(loaded_models[key]['config']['general']['train_valid_split'])
    
    if 1 < len(set(random_seed_list)):
        return False
    
    for i in range(len(loaded_models.keys()) - 1):
        if train_valid_split_list[i] != train_valid_split_list[i+1]:
            return False

    return True


# load model and predict
def predict_original_for_loader(loaded_models, dataloader, config, is_pseudo):
    
    predicts = defaultdict(lambda: {'original': defaultdict(dict), 'target': defaultdict(dict), 'processed': defaultdict(dict)})
    predicts_each = defaultdict()
    metrics = []

    for i, key in enumerate(loaded_models.keys()):
        print('【%d/%d】' % (i+1, len(loaded_models.keys())))
        model = loaded_models[key]['pseudo_model' if is_pseudo else 'model']
        for images, targets, image_ids in tqdm.tqdm(dataloader):
            image_id = image_ids[0]
            preds, _ = model(images, targets)
            metrics.append(calculate_score_for_each(preds, targets))

            if i == 0:
                predicts[image_id]['target']['boxes'] = targets[0]['boxes']
                predicts_each[image_id] = defaultdict(dict)
               
            predicts_each[image_id][key]['boxes'] = preds[0]['boxes']
            predicts_each[image_id][key]['scores'] = preds[0]['scores']
    
    for image_id, d in predicts_each.items():
        idx = 0
        all_empty = False
        while not all_empty:
            top_scores = []
            top_boxes = []
            for key in predicts_each[image_id].keys():
                if d[key]['scores'].shape[0] <= idx:
                    continue            
                top_scores.append(d[key]['scores'][idx])
                top_boxes.append(d[key]['boxes'][idx, :])  

            if len(top_scores) == 0:
                all_empty = True
                continue      

            top_scores = np.array(top_scores)
            top_boxes = np.array(top_boxes)
            top_sorted_idx = np.argsort(top_scores)[::-1]
            top_boxes = top_boxes[top_sorted_idx, :]
            top_scores = top_scores[top_sorted_idx]
            
            if 'boxes' not in predicts[image_id]['original']:
                predicts[image_id]['original']['boxes'] = top_boxes
                predicts[image_id]['original']['scores'] = top_scores
            else:
                if config['apply']:
                    keeped_top_score = predicts[image_id]['original']['scores'][-1]
                    top_scores -= np.max([0.0, (top_scores[0] - keeped_top_score + config['subtraction'])])
                predicts[image_id]['original']['boxes'] = np.concatenate([predicts[image_id]['original']['boxes'], top_boxes], axis=0)
                predicts[image_id]['original']['scores'] = np.concatenate([predicts[image_id]['original']['scores'], top_scores], axis=0)
                sorted_idx = np.argsort(predicts[image_id]['original']['scores'])[::-1]
                predicts[image_id]['original']['boxes'] = predicts[image_id]['original']['boxes'][sorted_idx, :]
                predicts[image_id]['original']['scores'] = predicts[image_id]['original']['scores'][sorted_idx]                        
            idx += 1
        
    return predicts, np.array(metrics)
    

def predict_original(loaded_models, train_data_loader, valid_data_loader, config, is_pseudo=False):

    print('predicting train...')
    train_predicts, train_metrics = predict_original_for_loader(loaded_models, train_data_loader, config, is_pseudo)
    print('predicting valid...')
    valid_predicts, valid_metrics = predict_original_for_loader(loaded_models, valid_data_loader, config, is_pseudo)

    mean_train_metrics = np.mean(train_metrics)
    mean_valid_metrics = np.mean(valid_metrics)
    print()
    print('%s train metrics: ' % ('pseudo' if is_pseudo else 'original'), mean_train_metrics)
    print('%s valid metrics: ' % ('pseudo' if is_pseudo else 'original'), mean_valid_metrics)
    return train_predicts, mean_train_metrics, valid_predicts, mean_valid_metrics

    
def apply_postprocess(train_predicts, valid_predicts, config):

        train_predicts = copy.deepcopy(train_predicts)
        valid_predicts = copy.deepcopy(valid_predicts)

        train_metrics = []
        valid_metrics = []
        for image_id in train_predicts.keys():
            processed = postprocessing([train_predicts[image_id]['original']], config)
            metrics = calculate_score_for_each(processed, [train_predicts[image_id]['target']])
            train_metrics.append(metrics)
            train_predicts[image_id]['processed']['boxes'] = processed[0]['boxes']
            train_predicts[image_id]['processed']['scores'] = processed[0]['scores']
        
        for image_id in valid_predicts.keys():
            processed = postprocessing([valid_predicts[image_id]['original']], config)
            metrics = calculate_score_for_each(processed, [valid_predicts[image_id]['target']])
            valid_metrics.append(metrics)
            valid_predicts[image_id]['processed']['boxes'] = processed[0]['boxes']
            valid_predicts[image_id]['processed']['scores'] = processed[0]['scores']
        
        train_metrics = np.array(train_metrics)
        valid_metrics = np.array(valid_metrics)

        return train_predicts, train_metrics, valid_predicts, valid_metrics


def load_models(predict_config):

    model_paths = predict_config['model_paths']
    loaded_models = defaultdict(dict)
    for path in model_paths:
        config_path = OUTPUT_DIR/path/'config.json'
        with open(str(config_path), 'r') as f:
            config = json.load(f)

        if 'kfold' in config['general'] and 0 < config['general']['kfold']:
            print('[WARN]: not supporting k-fold. Execute sys.exit().')
            sys.exit() 
        
        weight_path = list(sorted((OUTPUT_DIR/path).glob('*.pt')))[-1]  # latest weight
        loaded_models[path]['config'] = config
        loaded_models[path]['weight_path'] = weight_path
        model = Model(loaded_models[path]['config']['model'])
        model = model.load_state_dict(str(loaded_models[path]['weight_path'])).to(device)
        model.eval()

        # apply tta wrapper if needed
        if predict_config['test_time_augmentation']['apply']:
            print('apply TTA to %s.' % path)
            model = TTAModelWrapper(model, predict_config, device)

        loaded_models[path]['model'] = model

    general_config = loaded_models[model_paths[0]]['config']

    return loaded_models, general_config


def get_dataloader(general_config):

    def worker_init_fn(worker_id):   
        random.seed(worker_id+random_seed)   
        np.random.seed(worker_id+random_seed) 

    train_ids, valid_ids = train_valid_split(DATAFRAME, general_config)[0]
    train_ids = train_ids[:10] if debug else train_ids
    valid_ids = valid_ids[:2] if debug else valid_ids
    train_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(train_ids), :]
    valid_dataframe = DATAFRAME.loc[DATAFRAME['image_id'].isin(valid_ids), :]
    train_dataset = GWDDataset(train_dataframe, TRAIN_IMAGE_DIR, general_config, is_train=True, do_transform=False)
    valid_dataset = GWDDataset(valid_dataframe, TRAIN_IMAGE_DIR, general_config, is_train=False, do_transform=False)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, collate_fn=collate_fn)    
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, collate_fn=collate_fn)

    return train_ids, train_data_loader, valid_ids, valid_data_loader


def optimize_postprocess(postprocess_optimizer, predict_config=None):

    optimizer = {
        'nms': postprocess_optimizer.optimize_nms,
        'soft_nms': postprocess_optimizer.optimize_soft_nms,
        'wbf': postprocess_optimizer.optimize_wbf
    }


    for method in optimizer.keys():
        if not predict_config['optimize'][method]['apply']:
            continue 
        
        print()
        print('optimizing %s...' % method)

        best_params_dict = optimizer[method](n_calls=predict_config['optimize'][method]['n_calls'])
        config = get_config(name=method, **best_params_dict)
        
        opted_train_predicts, opted_train_metrics, opted_valid_predicts, opted_valid_metrics = apply_postprocess(postprocess_optimizer.train_predicts, postprocess_optimizer.valid_predicts, config)
        mean_opted_train_metrics = np.mean(opted_train_metrics)
        mean_opted_valid_metrics = np.mean(opted_valid_metrics)
        postprocess_optimizer.send(mean_opted_train_metrics, mean_opted_valid_metrics, method)
        print('%s best params: %s=%f, %s=%f' % (method, list(best_params_dict.keys())[0], list(best_params_dict.values())[0], list(best_params_dict.keys())[1], list(best_params_dict.values())[1]))
        print('%s train metrics: %f' % (method, mean_opted_train_metrics))
        print('%s valid metrics: %f' % (method, mean_opted_valid_metrics))


class TTAModelWrapper:

    def __init__(self, model, config, device):

        self.model = model
        self.device = device

        transform_list = {
            'original': [A.HorizontalFlip(p=0), ToTensorV2(p=1)],  # set `HorizontalFlip` p=0, this is only for avoid error.
            'hflip': [A.HorizontalFlip(p=1), ToTensorV2(p=1)],
            'vflip': [A.VerticalFlip(p=1), ToTensorV2(p=1)],
            'vhflip': [A.HorizontalFlip(p=1), A.VerticalFlip(p=1), ToTensorV2(p=1)]      
        }
        transform_inv_list = {
            'original': [A.HorizontalFlip(p=0), ToTensorV2(p=1)],
            'hflip': [A.HorizontalFlip(p=1), ToTensorV2(p=1)],
            'vflip': [A.VerticalFlip(p=1), ToTensorV2(p=1)],
            'vhflip': [A.HorizontalFlip(p=1), A.VerticalFlip(p=1), ToTensorV2(p=1)]      
        }
        using_tf_list =[k for k, v in config['test_time_augmentation']['augments'].items() if v]

        self.transforms = [
            A.Compose(transform_list[tf_name],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}) for tf_name in using_tf_list
        ]

        self.transforms_inv = [
            A.Compose(transform_inv_list[tf_name],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}) for tf_name in using_tf_list
        ]

    
    def __call__(self, images, dummy_targets):
        ts_images, ts_dummy_targets = self._transform(images, dummy_targets)
        outputs, _ = self.model(ts_images, ts_dummy_targets)
        outputs = [{k: v.data.cpu() for k, v in t.items()} for t in outputs]
        preds = self._transform_inv(images, outputs)
        return preds, {'loss': -1}

    def _transform(self, images, targets):
        """
        Input image and target should be only one. 
        Images and targets will be multi-lenght.
        """
        sample = {
            'image': images[0].permute(1, 2, 0).data.cpu().numpy(),
            'bboxes': targets[0]['boxes'],
            'labels': targets[0]['labels']
        }
        samples = [transform(**sample) for transform in self.transforms]
        ts_images = list(sample['image'].to(self.device) for sample in samples)
        return ts_images, targets * len(ts_images)

    def _transform_inv(self, images, outputs):
        samples = [
            {
            'image': images[0].permute(1, 2, 0).data.cpu().numpy(),
            'bboxes': output['boxes'],
            'labels': [1 for _ in output['boxes']]
            } for output in outputs
        ]

        boxes = []
        labels = []
        scores = list(itertools.chain.from_iterable([output['scores'].data.cpu().numpy() for output in outputs])) 

        outputs = [transform(**sample) for transform, sample in zip(self.transforms_inv, samples)]
        for idx in range(len(outputs)):
            boxes += outputs[idx]['bboxes']
            labels += [1 for _ in outputs[idx]['bboxes']]
        return [{'boxes': torch.tensor(boxes), 'scores': torch.tensor(scores), 'labels': torch.tensor(labels)}]


if __name__ == '__main__':

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('json_path')
    args = parser.parse_args()
    device = torch.device('cuda:0')

    assert os.path.exists(args.json_path), "json\'s name '%s' is not valid." % args.json_path

    with open(args.json_path, 'r') as f:
        predict_config = json.load(f)
    debug = predict_config['debug']

    SRC_DIR = Path('.').resolve()/'src'
    TRAIN_IMAGE_DIR = SRC_DIR/'train'
    TEST_IMAGE_DIR= SRC_DIR/'test'
    DATAFRAME = convert_dataframe(pd.read_csv(str(SRC_DIR/'train.csv')))    
    OUTPUT_DIR = Path('.').resolve()/'output'

    # load models
    # [WARN]: this general_config contains some information not to use
    loaded_models, general_config = load_models(predict_config)
    
    # check random_seed and train_valid_split
    assert sanity_check(loaded_models), 'The models you selected are invalid.'
    
    # set seed (not enough for complete reproducibility)
    random_seed = general_config['general']['seed']
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  
    torch.cuda.manual_seed(random_seed) 

    # get dataloader
    train_ids, train_data_loader, valid_ids, valid_data_loader = get_dataloader(general_config)

    # pseudo label training
    prepare_pseudo_labels(loaded_models, valid_data_loader, predict_config['pseudo_label'])
    retrain_with_pseudo_label(loaded_models, train_ids, valid_ids, TRAIN_IMAGE_DIR, DATAFRAME, predict_config['pseudo_label'])

    # predict train and valid 
    # ensemble if you selected multiple models
    train_predicts, train_metrics, valid_predicts, valid_metrics = predict_original(loaded_models, train_data_loader, valid_data_loader, predict_config["predict_normalization"])
    if predict_config['pseudo_label']['apply']:
        del train_predicts, train_metrics, valid_predicts, valid_metrics 
        train_predicts,  train_metrics,  valid_predicts, valid_metrics = predict_original(loaded_models, train_data_loader, valid_data_loader, predict_config["predict_normalization"], is_pseudo=True)

    # optimize postprocessing
    postprocess_optimizer = PostProcessOptimizer(train_predicts, train_metrics, valid_predicts, valid_metrics)
    optimize_postprocess(postprocess_optimizer, predict_config=predict_config)

    # log result
    postprocess_optimizer.log(predict_config, general_config)

    

    # # ここでサンプル描画
    # images, targets, image_ids = iter(train_data_loader).next()
    # image = (cv2.UMat(np.transpose(images[0].detach().cpu().numpy(), (1, 2, 0))).get() * 255).astype(np.uint8)
    # image_id = image_ids[0]
    # target_box = train_predicts[image_ids[0]]['target']['boxes'].detach().cpu().numpy()
    # for j in range(target_box.shape[0]):
    #     box = target_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
    
    # original_box = train_predicts[image_ids[0]]['original']['boxes']
    # for j in range(original_box.shape[0]):
    #     box = original_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 220, 0), 1)
    
    # processed_box = train_predicts[image_ids[0]]['processed']['boxes']
    # for j in range(processed_box.shape[0]):
    #     box = processed_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 3)
    # cv2.imwrite('./sample.png', image)

    # images, targets, image_ids = iter(valid_data_loader).next()
    # image = (cv2.UMat(np.transpose(images[0].detach().cpu().numpy(), (1, 2, 0))).get() * 255).astype(np.uint8)
    # image_id = image_ids[0]
    # target_box = valid_predicts[image_ids[0]]['target']['boxes'].detach().cpu().numpy()
    # for j in range(target_box.shape[0]):
    #     box = target_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
    
    # original_box = valid_predicts[image_ids[0]]['original']['boxes']
    # for j in range(original_box.shape[0]):
    #     box = original_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (220, 220, 0), 1)
    
    # processed_box = valid_predicts[image_ids[0]]['processed']['boxes']
    # for j in range(processed_box.shape[0]):
    #     box = processed_box[j]
    #     box = box.astype(np.int)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 3)
    # cv2.imwrite('./sample2.png', image)



