import logging
import torch

from cliff.command import Command

from cosmos.utils.glimpse_reader import read_glimpse

class Glimpse(Command):
    "A command to fit the data to the model"

    def get_parser(self, prog_name):
        """Command argument parsing."""
        parser = super(Glimpse, self).get_parser(prog_name)

        parser.add_argument("dataset", type=str,
                            help="Name of the dataset folder")

        parser.add_argument("-c", "--control", action="store_true",
                            help="Analyze control dataset")
        parser.add_argument("-dev", "--device", default="cpu",
                            type=str, metavar="\b",
                            help="GPU device")

        return parser

    def take_action(self, args):
        data = read_glimpse(args.dataset, D=14, dtype="test", device=args.device)
        data.save(args.dataset)
        if args.control:
            control = read_glimpse(args.dataset, D=14, dtype="control", device=args.device)
            control.save(args.dataset)
