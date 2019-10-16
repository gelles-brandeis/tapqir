import inspect
import os
import numpy as np
import pyro
# pyro & pytorch
import torch
from pyro import poutine
pyro.enable_validation(True)
# loading & saving data
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="classifier")
parser.add_argument("-m", "--model", default="v3", type=str)#,
                       # help="one of: {}".format(", ".join(sorted(models.keys()))))
parser.add_argument("-n", "--num-epochs", default=100, type=int)
parser.add_argument("-N", "--n-batch", default=16, type=int)
parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
#parser.add_argument("-F", "--f-batch", default=64, type=int)
parser.add_argument("-d", "--dataset", type=str)
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

from cosmos.models.FeatureExtraction import FeatureExtraction
from cosmos.models.FExtraction import FExtraction
from cosmos.models.JExtraction import JExtraction
from cosmos.models.JunkExtraction import JunkExtraction
#from models.Modelv1 import Modelv1
#from models.Modelv2 import Modelv2
from cosmos.models.Modelv3 import Modelv3
from cosmos.models.Modelv4 import Modelv4
from cosmos.models.Test import Test 
from cosmos.models.Modelv5 import Modelv5
from cosmos.models.Modelv6 import Modelv6
from cosmos.models.Modelv7 import Modelv7
#from cosmos.models.HMMv1 import HMMv1
models = dict()
models["feature"] = FeatureExtraction
models["feat"] = FExtraction
models["junk"] = JExtraction
#models["junk"] = JunkExtraction
#models["v1"] = Modelv1
#models["v2"] = Modelv2
models["v3"] = Modelv3
models["v4"] = Modelv4
models["test"] = Test
models["v5"] = Modelv5
models["v6"] = Modelv6
models["v7"] = Modelv7
#models["hmm"] = HMMv1

from cosmos.utils.aoi_reader import ReadAoi
from cosmos.utils.feature_read import ReadFeatures 

data = ReadAoi(args.dataset, device)
if args.model in ["feature", "feat"]:
    model = models[args.model](data, args.dataset, lr=args.learning_rate, jit=args.jit) # change here
elif args.model in ["junk"]:
    model = models[args.model](data, args.dataset, lr=args.learning_rate, jit=args.jit) # change here
else:
    data = ReadFeatures(data, args.dataset, device)
    model = models[args.model](data, args.dataset, K=2, lr=args.learning_rate, jit=args.jit) # change here


#print("classifying {}".format(args.dataset))
#model.fixed_epoch(args.n_batch, args.num_epochs)
