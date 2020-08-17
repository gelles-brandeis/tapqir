import sys

import os
import configparser
from cliff.command import Command

from PySide2.QtWidgets import QApplication
from cosmos.ui.qtgui import MainWindow

from cosmos.models.tracker import Tracker
from cosmos.models.spot import Spot


models = dict()
models["tracker"] = Tracker
models["spot"] = Spot


class Show(Command):
    "Show fitting results."

    def get_parser(self, prog_name):
        parser = super(Show, self).get_parser(prog_name)

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
        # read options.cfg fiel
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = os.path.join(args.dataset, "options.cfg")
        config.read(cfg_file)

        spot = args.spot or config["fit"].getint("spot")

        model = models[args.model](spot)
        app = QApplication(sys.argv)
        MainWindow(model, args.dataset, args.parameters)
        sys.exit(app.exec_())
