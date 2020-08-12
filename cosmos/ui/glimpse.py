from cliff.command import Command

from cosmos.utils.glimpse_reader import read_glimpse


class Glimpse(Command):
    """
    Read the data from glimpse files and save it in the CoSMoS native format.

    Path names to glimpse files can be provided by the dataset.cfg file, or
    alternatively, at the prompt input.

    """

    def get_parser(self, prog_name):
        parser = super(Glimpse, self).get_parser(prog_name)

        parser.add_argument("dataset", type=str,
                            help="Path to the dataset folder")

        return parser

    def take_action(self, args):
        read_glimpse(args.dataset, D=14)
