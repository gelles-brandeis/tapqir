import inspect
import os
import numpy as np
import pyro
# pyro & pytorch
import torch
pyro.enable_validation(True)
# loading & saving data
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="classifier")
parser.add_argument("-m", "--model", default="v3", type=str)#,
                       # help="one of: {}".format(", ".join(sorted(models.keys()))))
parser.add_argument("-n", "--num-epochs", type=int)
parser.add_argument("-N", "--n-batch", default=16, type=int)
parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-nc", "--negative-control", type=str)
parser.add_argument('--cuda0', action='store_true')
parser.add_argument('--cuda1', action='store_true')
parser.add_argument('--jit', action='store_true')
args = parser.parse_args()

if args.cuda0:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("using device: cuda{}".format(torch.cuda.current_device()))
elif args.cuda1:
    device = torch.device("cuda:1")
    torch.cuda.set_device(1)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("using device: cuda{}".format(torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("using device: cpu")

from cosmos.utils.aoi_reader import ReadAoi
from cosmos.models.features import Features 
from cosmos.models.detector import Detector
from cosmos.models.tracker import Tracker
models = dict()
models["features"] = Features
models["detector"] = Detector
models["tracker"] = Tracker

data = ReadAoi(args.dataset, device)
if args.negative_control:
    control = ReadAoi(args.negative_control, device)
else:
    control = None
model = models[args.model](data, control, K=2, lr=args.learning_rate, n_batch=args.n_batch, jit=args.jit) # change here

if args.num_epochs:
    model.load()
    model.epoch(args.num_epochs) 
