import sys

from cliff.command import Command

from PySide2.QtWidgets import QApplication
from cosmos.ui.qtgui import MainWindow


class Show(Command):
    "Show fitting results."

    def get_parser(self, prog_name):
        parser = super(Show, self).get_parser(prog_name)

        parser.add_argument("dataset", type=str,
                            help="Path to the dataset folder")
        parser.add_argument("parameters", type=str,
                            help="Path to the parameters folder")

        return parser

    def take_action(self, args):
        app = QApplication(sys.argv)
        MainWindow(args.dataset, args.parameters)
        sys.exit(app.exec_())
