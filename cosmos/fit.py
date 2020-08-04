import logging
import torch

from cliff.command import Command

from cosmos.utils.aoi_reader import ReadAoi
from cosmos.models.tracker import Tracker
from cosmos.models.marginal import Marginal
from cosmos.models.globalhwb import GlobalHWB
from cosmos.models.globalhw import GlobalHW


models = dict()
models["tracker"] = Tracker
models["marginal"] = Marginal
models["globalhwb"] = GlobalHWB
models["globalhw"] = GlobalHW

class Fit(Command):
    "A command to fit the data to the model"

    def get_parser(self, prog_name):
        """Command argument parsing."""
        parser = super(Fit, self).get_parser(prog_name)

        parser.add_argument("model", default="tracker", type=str,
                            help="Choose the model: {}".format(", ".join(models.keys())))
        parser.add_argument("dataset", type=str,
                            help="Name of the dataset folder")

        parser.add_argument("-it", "--num-iter", default=20000, type=int, metavar="\b",
                            help="Number of iterations")
        parser.add_argument("-bs", "--batch-size", default=8, type=int, metavar="\b",
                            help="Batch size")
        parser.add_argument("-lr", "--learning-rate", default=0.005, type=float, metavar="\b",
                            help="Learning rate")

        parser.add_argument("-c", "--control", action="store_true",
                            help="Analyze control dataset")
        parser.add_argument("-dev", "--device", default="cuda",
                            type=str, metavar="\b",
                            help="GPU device")
        parser.add_argument("--jit", action="store_true",
                            help="Just in time compilation")

        return parser

    def take_action(self, args):
        model = models[args.model]()
        model.load(args.dataset, args.control, args.device)
        model.settings(args.learning_rate, args.batch_size)
        model.run(args.num_iter)
        """
        logger.info("Using device: {}".format(args.device))
        logger.info("Loading dataset: {}".format(args.dataset))

        logger.info("Model: {}".format(args.model))
        """
