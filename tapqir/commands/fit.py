from cliff.command import Command

import os
import torch
import configparser
from tapqir.models import models


class Fit(Command):
    r"""
    Fit the data to the selected model.

    Command options are read from the ``options.cfg`` file and will be
    overwritten if provided explicitly.

    Output files of the analysis are saved in ``runs/model/version/S/control/lr/bs/``:

    - ``params`` and ``optimizer`` are model parameters and optimizer state saved in PyTorch format
    - ``global_params.csv`` contains values of global parameters
    - ``parameters.mat`` contains MAP estimate of parameters in MATLAB format
    - ``scalar`` folder containing global parameter values over iterations
    - ``run.log`` log file
    """

    def get_parser(self, prog_name):
        parser = super(Fit, self).get_parser(prog_name)

        parser.add_argument("model", default="spotdetection", type=str,
                            help="Available models: {}".format(", ".join(models.keys())))
        parser.add_argument("dataset_path", default=".", type=str,
                            help="Path to the dataset folder")

        parser.add_argument("-s", metavar="NUM_STATES", type=int,
                            help="Number of molecular states (default: 1)")
        parser.add_argument("-k", metavar="K_MAX", type=int,
                            help="Maximum number of spots (default: 2)")
        parser.add_argument("-it", metavar="NUM_ITER", type=int,
                            help="Number of iterations (default: 30000)")
        parser.add_argument("-infer", metavar="INFER", type=int,
                            help="Number of inference iterations (default: 20000)")
        parser.add_argument("-bs", metavar="BATCH_SIZE", type=int,
                            help="Batch size (default: 5)")
        parser.add_argument("-lr", metavar="LEARNING_RATE", type=float,
                            help="Learning rate (default: 0.005)")

        parser.add_argument("-c", metavar="CONTROL", type=bool,
                            help="Analyze control dataset (default: False)")
        parser.add_argument("-dev", metavar="DEVICE", type=str,
                            help="Compute device (default: cuda)")

        return parser

    def take_action(self, args):
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = os.path.join(args.dataset_path, "options.cfg")
        config.read(cfg_file)

        states = args.s or config["fit"].getint("num_states")
        k_max = args.k or config["fit"].getint("k_max")
        num_iter = args.it or config["fit"].getint("num_iter")
        infer = args.infer or config["fit"].getint("infer")
        batch_size = args.bs or config["fit"].getint("batch_size")
        learning_rate = args.lr or config["fit"].getfloat("learning_rate")
        control = args.c or config["fit"].getboolean("control")
        device = args.dev or config["fit"].get("device")

        if device == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        model = models[args.model](states, k_max)
        model.load(args.dataset_path, control, device)

        model.settings(learning_rate, batch_size)
        if batch_size == 0:
            # add new batch_size to options.cfg
            config.set("fit", "batch_size", str(model.batch_size))
            with open(cfg_file, "w") as configfile:
                config.write(configfile)
        model.run(num_iter, infer)
