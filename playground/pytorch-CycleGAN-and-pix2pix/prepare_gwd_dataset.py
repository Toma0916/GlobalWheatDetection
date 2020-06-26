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

sys.path.append('./../../utils')

import numpy as np 
from numpy.random.mtrand import RandomState
import pandas as pd 

from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import cv2

import matplotlib.pyplot as plt

source_domain = 'ethz_1'
target_domain = 'arvalis_1'

from functions import convert_dataframe


if __name__ == '__main__':
    SRC_DIR = Path('./../../').resolve()/'src'
    TRAIN_IMAGE_DIR = SRC_DIR/'train'
    TEST_IMAGE_DIR= SRC_DIR/'test'
    DATAFRAME = convert_dataframe(pd.read_csv(str(SRC_DIR/'train.csv')))

    DATASET_DIR = Path('.').resolve()/'datasets'/'gwd_style_transfer'

    source_ids = DATAFRAME.loc[DATAFRAME['source']==source_domain, 'image_id'].unique()
    target_ids = DATAFRAME.loc[DATAFRAME['source']==target_domain, 'image_id'].unique()
    source_num = source_ids.shape[0]
    source_train_num = int(source_num * 0.8)
    target_num = target_ids.shape[0]
    target_train_num = int(target_num * 0.8)
    
    source_train_ids = source_ids[:source_train_num]
    source_test_ids = source_ids[source_train_num:]
    target_train_ids = target_ids[:target_train_num]
    target_test_ids = target_ids[target_train_num:]


    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    os.mkdir(DATASET_DIR)
    os.mkdir(DATASET_DIR/'trainA')
    os.mkdir(DATASET_DIR/'trainB')
    os.mkdir(DATASET_DIR/'testA')
    os.mkdir(DATASET_DIR/'testB')

    for id in source_train_ids:
        shutil.copy(TRAIN_IMAGE_DIR/('%s.jpg' % id), DATASET_DIR/'trainA')
    for id in source_test_ids:
        shutil.copy(TRAIN_IMAGE_DIR/('%s.jpg' % id), DATASET_DIR/'testA')
    for id in target_train_ids:
        shutil.copy(TRAIN_IMAGE_DIR/('%s.jpg' % id), DATASET_DIR/'trainB')
    for id in target_test_ids:
        shutil.copy(TRAIN_IMAGE_DIR/('%s.jpg' % id), DATASET_DIR/'testB')

# python -m visdom.server
# python train.py --dataroot ./datasets/gwd_style_transfer --name gwd_style_transfer --model cycle_gan 

