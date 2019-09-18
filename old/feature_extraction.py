import inspect
import os
import argparse
import numpy as np
import pyro
# pyro & pytorch
import torch
from pyro import poutine
# loading & saving data
from tqdm import tqdm

parser = argparse.ArgumentParser(description="feature extraction")
#parser.add_argument("-m", "--model", default="1", type=str,
                        #help="one of: {}".format(", ".join(sorted(models.keys()))))
parser.add_argument("-n", "--num-epochs", default=50, type=int)
parser.add_argument("-N", "--n-batch", default=16, type=int)
parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

if args.cuda:
    #device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from noise import GammaNoise
from FeatureExtraction import FeatureExtraction 

from aoi_reader import ReadAoi

data = ReadAoi(args.dataset)
# init
model_name = "feature-extraction"
Model = type(model_name, (FeatureExtraction, GammaNoise),{}) # change here
model = Model(data, args.learning_rate) # change here

print("extracting features for {}".format(args.dataset))
model.epoch(args.n_batch, args.num_epochs)

model.save()
