import os
from pyro import poutine
from pyro.ops.stats import pi
from scipy.io import savemat
import torch
import configparser
from cliff.command import Command

from cosmos.models import models


class Matlab(Command):
    "Convert parameters into matlab format."

    def get_parser(self, prog_name):
        parser = super(Matlab, self).get_parser(prog_name)

        parser.add_argument("model", default="tracker", type=str,
                            help="Choose the model: {}".format(", ".join(models.keys())))
        parser.add_argument("dataset", type=str,
                            help="Path to the dataset folder")
        parser.add_argument("parameters", type=str,
                            help="Path to the parameters folder")

        parser.add_argument("-s", "--spot", type=int, metavar="\b",
                            help="Max. number of molecules per spot")

        return parser

    def take_action(self, args):
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = os.path.join(args.dataset, "options.cfg")
        config.read(cfg_file)

        spot = args.spot or config["fit"].getint("spot")

        model = models[args.model](spot)
        model.load(args.dataset, False, "cpu")
        model.load_parameters(args.parameters)
        model.n = torch.arange(model.data.N)
        model.frames = torch.arange(model.data.F)
        trace = poutine.trace(model.guide).get_trace()

        items = ["z", "height_0", "height_1", "width_0", "width_1",
                      "x_0", "x_1", "y_0", "y_1", "background"]
        params_dict = {}
        params_dict["aoilist"] = model.data.target.index.values
        params_dict["framelist"] = model.data.drift.index.values
        for p in items:
            if p == "z":
                params_dict[f"{p}_probs"] = model.predictions["z_prob"]
                params_dict[f"{p}_binary"] = model.predictions["z"]
            else:
                hpd = pi(trace.nodes[f"d/{p}"]["fn"].sample((500,)).data.squeeze().cpu(), 0.95, dim=0)
                mean = trace.nodes[f"d/{p}"]["fn"].mean.data.squeeze().cpu()
                params_dict[f"{p}_high"] = hpd[0].numpy()
                params_dict[f"{p}_low"] = hpd[1].numpy()
                params_dict[f"{p}_mean"] = mean.numpy()
        savemat(os.path.join(args.parameters, "params.mat"), params_dict)
