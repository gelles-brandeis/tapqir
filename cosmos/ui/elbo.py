import os
import pyro
import torch
import numpy as np
from tqdm import tqdm
import configparser
from cliff.command import Command

from cosmos.models import Spot, Tracker


models = dict()
models["tracker"] = Tracker
models["spot"] = Spot


class ELBO(Command):
    "Calculate ELBO estimate"

    def get_parser(self, prog_name):
        parser = super(ELBO, self).get_parser(prog_name)

        parser.add_argument("model", default="tracker", type=str,
                            help="Choose the model: {}".format(", ".join(models.keys())))
        parser.add_argument("dataset", type=str,
                            help="Path to the dataset folder")

        parser.add_argument("-s", "--spot", type=int, metavar="\b",
                            help="Max. number of molecules per spot")
        parser.add_argument("-it", "--num-iter", type=int, metavar="\b",
                            help="Number of iterations")
        parser.add_argument("-bs", "--batch-size", type=int, metavar="\b",
                            help="Batch size")
        parser.add_argument("-lr", "--learning-rate", type=float, metavar="\b",
                            help="Learning rate")

        parser.add_argument("-c", "--control", type=bool,
                            help="Analyze control dataset")
        parser.add_argument("-dev", "--device", type=str, metavar="\b",
                            help="Compute device")

        return parser

    def take_action(self, args):
        # read options.cfg fiel
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = os.path.join(args.dataset, "options.cfg")
        config.read(cfg_file)

        spot = args.spot or config["fit"].getint("spot")
        num_iter = args.num_iter or config["fit"].getint("num_iter")
        batch_size = args.batch_size or config["fit"].getint("batch_size")
        learning_rate = args.learning_rate or config["fit"].getfloat("learning_rate")
        control = args.control or config["fit"].getboolean("control")
        device = args.device or config["fit"].get("device")

        if device == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        model = models[args.model](spot)
        model.load(args.dataset, control, device)
        model.settings(learning_rate, batch_size)
        losses = []
        pyro.set_rng_seed(num_iter)
        for i in tqdm(range(num_iter)):
            losses.append(model.svi.evaluate_loss())
        np.save(os.path.join(model.path, "elbo.npy"), np.array(losses))
        print("mean: {:11.1f}".format(np.mean(losses)))
        print("std:  {:11.1f}".format(np.std(losses) / np.sqrt(num_iter)))
