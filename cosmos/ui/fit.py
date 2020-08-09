from cliff.command import Command

from cosmos.models.tracker import Tracker
from cosmos.models.globalhw import GlobalHW


models = dict()
models["tracker"] = Tracker
models["globalhw"] = GlobalHW


class Fit(Command):
    "Fit the data to the model"

    def get_parser(self, prog_name):
        parser = super(Fit, self).get_parser(prog_name)

        parser.add_argument("model", default="tracker", type=str,
                            help="Choose the model: {}".format(", ".join(models.keys())))
        parser.add_argument("dataset", type=str,
                            help="Path to the dataset folder")

        parser.add_argument("-it", "--num-iter", default=20000, type=int, metavar="\b",
                            help="Number of iterations")
        parser.add_argument("-bs", "--batch-size", default=8, type=int, metavar="\b",
                            help="Batch size")
        parser.add_argument("-lr", "--learning-rate", default=0.005, type=float, metavar="\b",
                            help="Learning rate")

        parser.add_argument("-c", "--control", action="store_true",
                            help="Analyze control dataset")
        parser.add_argument("-dev", "--device", default="cuda",
                            type=str, metavar="\b",
                            help="GPU device")
        parser.add_argument("--jit", action="store_true",
                            help="Just in time compilation")

        return parser

    def take_action(self, args):
        model = models[args.model]()
        model.load(args.dataset, args.control, args.device)
        model.settings(args.learning_rate, args.batch_size)
        model.run(args.num_iter)
