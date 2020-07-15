import argparse
import cosmos
import pyro
import torch
import logging
import subprocess
import os

from cosmos.utils.aoi_reader import ReadAoi
from cosmos.models.tracker import Tracker
from cosmos.models.marginal import Marginal
from cosmos.models.hmm import HMM
from cosmos.models.feature import Feature
from cosmos.models.globalhwb import GlobalHWB

pyro.enable_validation(True)

models = dict()
models["tracker"] = Tracker
models["marginal"] = Marginal
models["hmm"] = HMM
models["feature"] = Feature
models["globalhwb"] = GlobalHWB

# parse command-line arguments and execute the main method
parser = argparse.ArgumentParser(
    description="cosmos {}. Bayesian analysis of single molecule image data" \
                .format(cosmos.__version__),
    #usage="%(prog)s [OPTIONS]",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    add_help=False)
    #epilog="Good luck!")
subparsers = parser.add_subparsers(dest="subparser_name")

parser_run = subparsers.add_parser("run")
group_run = parser_run.add_argument_group("options")
group2 = parser.add_argument_group("other arguments")


parser_run.add_argument("model", default="tracker", type=str,
                    help="Choose the model: {}".format(", ".join(models.keys())))
parser_run.add_argument("dataset", type=str,
                    help="Name of the dataset folder")

group_run.add_argument("-it", "--num-iter", default=20000, type=int, metavar="\b",
                    help="Number of iterations")
group_run.add_argument("-bs", "--batch-size", default=8, type=int, metavar="\b",
                    help="Batch size")
group_run.add_argument("-lr", "--learning-rate", default=0.005, type=float, metavar="\b",
                    help="Learning rate")

group_run.add_argument("-c", "--control", action="store_true",
                    help="Analyze control dataset")
group_run.add_argument("-dev", "--device", default="cuda0",
                    choices=["cuda0", "cuda1", "cpu"], type=str, metavar="\b",
                    help="GPU device")
group_run.add_argument("--jit", action="store_true",
                    help="Just in time compilation")

parser_view = subparsers.add_parser("view")
#parser_view.add_argument("--results", action="store_false",
#                    help="Show results")

group2.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                    help="show this help message.")
group2.add_argument("-v", "--version", action="version",
                    version="%(prog)s {}".format(cosmos.__version__),
                    help="show program's version number.")
args = parser.parse_args()


# setup and training
def main(args=args):
    if args.subparser_name == "view":
        subprocess.run("voila {}".format(os.path.join(cosmos.__path__[0], "app.ipynb")),
            shell=True)
        return
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
    data, control = ReadAoi(args.dataset, args.control, device)

    logger.info("Model: {}".format(args.model))
    model = models[args.model](
            data, control, path=args.dataset, K=2, lr=args.learning_rate,
            n_batch=args.batch_size, jit=args.jit)

    if args.num_iter:
        model.load_checkpoint()
        model.train(args.num_iter)
