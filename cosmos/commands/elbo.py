import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import configparser
from cliff.command import Command

from cosmos.models import models


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

        params_last = pd.read_csv(
            os.path.join(model.path, "params_last.csv"),
            header=None, squeeze=True, index_col=0)
        if "ELBO_mean" not in params_last.index:
            losses = []
            for i in tqdm(range(1000)):
                elbo = model.svi.evaluate_loss()
                if not np.isnan(elbo):
                    losses.append(elbo)
            params_last["ELBO_mean"] = np.mean(losses)
            params_last["ELBO_std"] = np.std(losses) / np.sqrt(len(losses))
            params_last["ELBO_var"] = np.var(losses) / len(losses)
            params_last["elbo_var"] = np.var(losses)
            params_last.to_csv(os.path.join(model.path, "params_last.csv"))
        print(params_last["ELBO_mean"])
        print(params_last["ELBO_std"])
        losses = []
        for i in tqdm(range(num_iter)):
            elbo = model.svi.evaluate_loss()
            if not np.isnan(elbo):
                losses.append(elbo)

        params_last["ELBO_mean"] = (
            (params_last["ELBO_mean"] / params_last["ELBO_var"] + np.sum(losses) / params_last["elbo_var"])
            / (1 / params_last["ELBO_var"] + len(losses) / params_last["elbo_var"]))
        params_last["ELBO_var"] = 1 / (1 / params_last["ELBO_var"] + len(losses) / params_last["elbo_var"])
        params_last["ELBO_std"] = np.sqrt(params_last["ELBO_var"])
        print(params_last["ELBO_mean"])
        print(params_last["ELBO_std"])
        print(len(losses))
        params_last.to_csv(os.path.join(model.path, "params_last.csv"))
