import sys

from cliff.command import Command

from PySide2.QtWidgets import QApplication
from tapqir.commands.qtgui import MainWindow

from tapqir.models import models


class Show(Command):
    "Show fitting results."

    def get_parser(self, prog_name):
        parser = super(Show, self).get_parser(prog_name)

        parser.add_argument("model", default="spotdetection", type=str,
                            help="Available models: {}".format(", ".join(models.keys())))
        parser.add_argument("dataset_path", default=".", type=str,
                            help="Path to the dataset folder")
        parser.add_argument("parameters_path", type=str,
                            help="Path to the parameters folder")

        return parser

    def take_action(self, args):
        model = models[args.model](1, 2)
        app = QApplication(sys.argv)
        MainWindow(model, args.dataset_path, args.parameters_path)
        sys.exit(app.exec_())
