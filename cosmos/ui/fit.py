from cliff.command import Command

import os
import torch
import configparser
from cosmos.models.tracker import Tracker


models = dict()
models["tracker"] = Tracker


class Fit(Command):
    "Fit the data to the model"

    def get_parser(self, prog_name):
        parser = super(Fit, self).get_parser(prog_name)

        parser.add_argument("model", default="tracker", type=str,
                            help="Choose the model: {}".format(", ".join(models.keys())))
        parser.add_argument("dataset", default=".", type=str,
                            help="Path to the dataset folder")

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
        parser.add_argument("--jit", action="store_true",
                            help="Just in time compilation")

        return parser

    def take_action(self, args):
        # read options.cfg fiel
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = os.path.join(args.dataset, "options.cfg")
        config.read(cfg_file)

        num_iter = args.num_iter or config["fit"].getint("num_iter")
        batch_size = args.batch_size or config["fit"].getint("batch_size")
        learning_rate = args.learning_rate or config["fit"].getfloat("learning_rate")
        control = args.control or config["fit"].getboolean("control")
        device = args.device or config["fit"].get("device")

        if device == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        model = models[args.model]()
        model.load(args.dataset, control, device)
        model.settings(learning_rate, batch_size)
        model.run(num_iter)
