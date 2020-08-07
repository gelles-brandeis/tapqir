import logging
import torch

from cliff.command import Command

from cosmos.utils.glimpse_reader import read_glimpse

class Glimpse(Command):
    """
    Read the data from glimpse files and save it in the CoSMoS native fromat.

    Path names to glimpse files can be provided by the dataset.cfg file, or
    alternatively, at the prompt input.

    """

    def get_parser(self, prog_name):
        parser = super(Glimpse, self).get_parser(prog_name)

        parser.add_argument("dataset", type=str,
                            help="Path to the dataset folder")

        parser.add_argument("-c", "--control", action="store_true",
                            help="Read and save control data as well")

        return parser

    def take_action(self, args):
        data = read_glimpse(args.dataset, D=14, dtype="test")
        data.save(args.dataset)
        if args.control:
            control = read_glimpse(args.dataset, D=14, dtype="control")
            control.save(args.dataset)
