import configparser
import sys
from pathlib import Path

from cliff.command import Command
from pyroapi import pyro_backend
from PySide2.QtWidgets import QApplication

from tapqir.commands.qtgui import MainWindow
from tapqir.models import models


class Show(Command):
    "Show fitting results."

    def get_parser(self, prog_name):
        parser = super(Show, self).get_parser(prog_name)

        parser.add_argument(
            "model",
            default="cosmos",
            type=str,
            help="Available models: {}".format(", ".join(models.keys())),
        )
        parser.add_argument(
            "dataset_path", default=".", type=str, help="Path to the dataset folder"
        )
        parser.add_argument(
            "parameters_path", type=str, help="Path to the parameters folder"
        )
        parser.add_argument(
            "-backend", metavar="BACKEND", type=str, help="Pyro backend (default: pyro)"
        )

        return parser

    def take_action(self, args):
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = Path(args.dataset_path) / "options.cfg"
        config.read(cfg_file)

        backend = args.backend or config["fit"].get("backend")
        # pyro backend
        if backend == "pyro":
            PYRO_BACKEND = "pyro"
        elif backend == "funsor":
            PYRO_BACKEND = "contrib.funsor"
        else:
            raise ValueError("Only pyro and funsor backends are supported.")

        with pyro_backend(PYRO_BACKEND):
            model = models[args.model](1, 2)
            app = QApplication(sys.argv)
            MainWindow(model, args.dataset_path, args.parameters_path)
            sys.exit(app.exec_())
