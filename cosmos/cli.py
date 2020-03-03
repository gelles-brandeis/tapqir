import pyro
import torch
import argparse
import logging

from cosmos.utils.aoi_reader import ReadAoi
from cosmos.models.tracker import Tracker
from cosmos.models.marginal import Marginal

pyro.enable_validation(True)

models = dict()
models["tracker"] = Tracker
models["marginal"] = Marginal

# parse command-line arguments and execute the main method
parser = argparse.ArgumentParser(
    description="Bayesian analysis of single molecule image data",
    usage="%(prog)s --model=Model --dataset=Dataset \
    --negative-control=NegativeControl --num-steps=1000 \
    --learning-rate=0.001 --n-batch=32 --device=cuda0",
    epilog="Good luck!")
parser.add_argument("-m", "--model", default="features",
                    choices=models.keys(), type=str)
parser.add_argument("-n", "--num-steps", type=int)
parser.add_argument("-N", "--n-batch", default=32, type=int)
parser.add_argument("-lr", "--learning-rate", default=0.001, type=float)
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-nc", "--negative-control", type=str)
parser.add_argument("-dev", "--device",
                    default="cuda0", choices=["cuda0", "cuda1", "cpu"],
                    type=str)
parser.add_argument("--jit", action="store_true")
parser.add_argument("--sample", action="store_true")
args = parser.parse_args()


# setup and training
def main(args=args):
    # create logger with args.log
    logger = logging.getLogger("cosmos")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p")
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

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
    data, control = ReadAoi(args.dataset, args.negative_control, device)

    logger.info("Model: {}".format(args.model))
    model = models[args.model](
            data, control, K=2, lr=args.learning_rate,
            n_batch=args.n_batch, jit=args.jit)

    if args.num_steps and not args.sample:
        model.load_checkpoint()
        model.train(args.num_steps)
