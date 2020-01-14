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


models = dict()
models["tracker"] = Tracker

# setup and training
def main(args):
    # create logger with args.log
    logger = logging.getLogger("cosmos")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler("{}.log".format(args.dataset))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Created logfile: {}.log".format(args.dataset))

    if args.device == "cuda0":
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    elif args.device == "cuda1":
        device = torch.device("cuda:1")
        torch.cuda.set_device(1)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    elif args.device == "cpu":
        device = torch.device("cpu")
    logger.info("Using device: {}".format(args.device))

    logger.info("Loading dataset: {}".format(args.dataset))
    data = ReadAoi(args.dataset, device)
    if args.negative_control:
        logger.info("Loading dataset: {}".format(args.negative_control))
        control = ReadAoi(args.negative_control, device)
    else:
        control = None

    logger.info("Model: {}".format(args.model))
    model = models[args.model](data, control, K=2, lr=args.learning_rate, n_batch=args.n_batch, jit=args.jit) # change here

    if args.sample:
        model.sample()

    if args.num_steps and not args.sample:
        model.load_checkpoint()
        model.train(args.num_steps) 

# parse command-line arguments and execute the main method
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayesian analysis of single molecule image data",
                                     usage="python %(prog)s --model=Model --dataset=Dataset --negative-control=NegativeControl --num-steps=1000 --learning-rate=0.001 --n-batch=32 --device=cuda0",
                                     epilog="Good luck!")
    parser.add_argument("-m", "--model", default="features", choices=models.keys(), type=str)
    parser.add_argument("-n", "--num-steps", type=int)
    parser.add_argument("-N", "--n-batch", default=32, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-nc", "--negative-control", type=str)
    parser.add_argument("-dev", "--device", default="cuda0", choices=["cuda0", "cuda1", "cpu"], type=str)
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()

    main(args)
