import logging
import sys

from cliff.command import Command

from PySyde2.QtWidgets import QApplication
from cosmos.ui.qtgui import MainWindow

class Show(Command):
    "A command to fit the data to the model"

    def get_parser(self, prog_name):
        """Command argument parsing."""
        parser = super(Show, self).get_parser(prog_name)

        parser.add_argument("dataset", type=str,
                            help="Name of the dataset folder")
        parser.add_argument("parameters", type=str,
                            help="Name of the parameters folder")

        parser.add_argument("-c", "--control", action="store_true",
                            help="Analyze control dataset")
        parser.add_argument("-dev", "--device", default="cpu",
                            type=str, metavar="\b",
                            help="GPU device")

        return parser

    def take_action(self, args):
        app = QApplication(sys.argv)
        window = MainWindow(
            args.dataset, args.parameters,
            args.control, args.device
        )
        sys.exit(app.exec_())
