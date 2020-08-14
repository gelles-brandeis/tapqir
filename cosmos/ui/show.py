import sys

from cliff.command import Command

from PySide2.QtWidgets import QApplication
from cosmos.ui.qtgui import MainWindow

from cosmos.models.tracker import Tracker
from cosmos.models.globalhw import GlobalHW
from cosmos.models.marginal import Marginal


models = dict()
models["tracker"] = Tracker
models["globalhw"] = GlobalHW
models["marginal"] = Marginal


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

        return parser

    def take_action(self, args):
        model = models[args.model]()
        app = QApplication(sys.argv)
        MainWindow(model, args.dataset, args.parameters)
        sys.exit(app.exec_())
