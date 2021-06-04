# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
from pathlib import Path

from cliff.command import Command
from pyroapi import pyro_backend

from tapqir.models import models
from tapqir.utils.stats import save_stats


class Save(Command):
    """
    Compute 95% confidence intervals for model parameters.
    """

    def get_parser(self, prog_name):
        parser = super(Save, self).get_parser(prog_name)

        parser.add_argument(
            "model",
            default="cosmos",
            type=str,
            help="Available models: {}".format(", ".join(models.keys())),
        )
        parser.add_argument(
            "path", default=".", type=str, help="Path to the dataset folder"
        )
        parser.add_argument(
            "-backend", metavar="BACKEND", type=str, help="Pyro backend (default: pyro)"
        )

        parser.add_argument(
            "-dev", metavar="DEVICE", type=str, help="Compute device (default: cuda)"
        )
        parser.add_argument(
            "-dtype",
            metavar="DTYPE",
            type=str,
            help="Floating number precision (default: float32)",
        )
        parser.add_argument(
            "--matlab",
            action="store_true",
            help="Save parameters in matlab format (default: False)",
        )

        return parser

    def take_action(self, args):
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = Path(args.path) / "options.cfg"
        config.read(cfg_file)

        backend = args.backend or config["fit"].get("backend")
        dtype = args.dtype or config["fit"].get("dtype")

        # pyro backend
        if backend == "pyro":
            PYRO_BACKEND = "pyro"
        elif backend == "funsor":
            import funsor
            import pyro.contrib.funsor  # noqa: F401

            funsor.set_backend("torch")
            PYRO_BACKEND = "contrib.funsor"
        else:
            raise ValueError("Only pyro and funsor backends are supported.")

        with pyro_backend(PYRO_BACKEND):

            model = models[args.model](1, 2, "cpu", dtype)
            model.load(args.path)
            model.load_checkpoint(param_only=True)

            save_stats(model, args.path, save_matlab=args.matlab)
