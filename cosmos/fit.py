import logging
import torch

from cliff.command import Command

from cosmos.utils.aoi_reader import ReadAoi
from cosmos.models.tracker import Tracker
from cosmos.models.marginal import Marginal
from cosmos.models.globalhwb import GlobalHWB


models = dict()
models["tracker"] = Tracker
models["marginal"] = Marginal
models["globalhwb"] = GlobalHWB

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
        device = torch.device(args.device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        data, control = ReadAoi(args.dataset, args.control, device)

        model = models[args.model](
                data, control, path=args.dataset, K=2, lr=args.learning_rate,
                n_batch=args.batch_size, jit=args.jit, device=device)

        if args.num_iter:
            model.load_checkpoint()
            model.train(args.num_iter)
        """
        logger.info("Using device: {}".format(args.device))
        logger.info("Loading dataset: {}".format(args.dataset))

        logger.info("Model: {}".format(args.model))
        """
