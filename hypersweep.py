import argparse
import optuna
from optuna.samplers import TPESampler
import json
import requests
from pathlib import Path
import datetime
import os
import time
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from natsort import natsorted
import inspect
import sys
import pickle
import shutil
import copy
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
target_dir = os.path.join(current_dir, 'cocoapi', 'PythonAPI')
sys.path.insert(0, target_dir)

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups

from engine import train_one_epoch, evaluate
import utils
import transforms as T

import mlflow
import mlflow.pytorch
mlflow.set_tracking_uri('http://mlflow.172.26.62.216.nip.io')

from torch.utils.tensorboard import SummaryWriter
from auto_train import *


def objective(trial, args):
    
    # fixed hyperparams
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, str(trial.number))
    args.name = str(trial.number)
    
    # hyperparams to be optimized
    # args.model =     # model switching will follow later
    args.lr = trial.suggest_discrete_uniform('lr', 0.001, 0.015, 0.001)
    args.lr_gamma = trial.suggest_loguniform('lr_gamma', 0.005, 0.5)
    args.lr_step_size = trial.suggest_discrete_uniform('lr_step_size', 3, 24, 1)
    args.momentum = trial.suggest_uniform('momentum', 0, 0.99)
    args.weight_decay = trial.suggest_discrete_uniform('weight_decay', 0.0001, 0.0005, 0.0001)
    
    metric = main(args, trial)
    
    return metric

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--name', default='test', help='name of run')
    parser.add_argument('--experiment_name', default='new', help='name of experiment for mlflow (and study for optuna)')
    parser.add_argument('--data-path', default='dataset', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--early_stopping', default=False, help='use early stopping')
    parser.add_argument('--early_stopping_num_epochs', default=10, type=int, help='how many epochs to wait for improvement')
    parser.add_argument('--stopping_metric', default='mAP', help='metric to use for early stopping calculation: mAP, mAP0.5, mAR, mAR0.5')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu '
                        '::: 0.02/8*$NGPU')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=7, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--num_checkpoints', default=10, type=int, help='number of checkpoints to keep during training. Keeps memory requirements in check')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    args = parser.parse_args()
    
    storage='postgresql://mlflow_user:mlflow@172.26.62.216:5432/optuna'
    mlflow.set_experiment(args.experiment_name)
    study = optuna.load_study(study_name=args.experiment_name, storage=storage)
    study.optimize(lambda trial: objective(trial, args), n_trials=20, catch=(RuntimeError,))