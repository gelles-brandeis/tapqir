import inspect
import os
import numpy as np
import pyro
import torch
pyro.enable_validation(True)
from tqdm import tqdm
import argparse
import logging
import configparser

from cosmos.utils.aoi_reader import ReadAoi
from cosmos.models.tracker import Tracker

#log = logging.getLogger()

models = dict()
models["tracker"] = Tracker

# setup and training
def main(args):

    if args.device == "cuda0":
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif args.device == "cuda1":
        device = torch.device("cuda:1")
        torch.cuda.set_device(1)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif args.device == "cpu":
        device = torch.device("cpu")
    #logging.info("Using device: {}".format(args.device))
    #logging.info("Using device: cuda{}".format(torch.cuda.current_device()))

    data = ReadAoi(args.dataset, device)
    if args.negative_control:
        control = ReadAoi(args.negative_control, device)
    else:
        control = None
    model = models[args.model](data, control, K=2, lr=args.learning_rate, n_batch=args.n_batch, jit=args.jit) # change here
    data.log.info("Model: {}".format(args.model))

    if args.sample:
        model.sample()

    if args.num_iter and not args.sample:
        model.load_checkpoint()
        model.train(args.num_iter) 

# parse command-line arguments and execute the main method
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayesian analysis of single molecule image data",
                                     usage="python %(prog)s --model=Model --dataset=Dataset --negative-control=NegativeControl --num-iter=1000 --learning-rate=0.001 --n-batch=32 --device=cuda0",
                                     epilog="Good luck!")
    parser.add_argument("-m", "--model", default="features", choices=models.keys(), type=str)
    parser.add_argument("-n", "--num-iter", type=int)
    parser.add_argument("-N", "--n-batch", default=32, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-nc", "--negative-control", type=str)
    parser.add_argument("-dev", "--device", default="cuda0", choices=["cuda0", "cuda1", "cpu"], type=str)
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()

    main(args)
