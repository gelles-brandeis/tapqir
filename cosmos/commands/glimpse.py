from cliff.command import Command

from cosmos.utils.glimpse_reader import read_glimpse


class Glimpse(Command):
    """
    Read the data from glimpse files and save it in the Cosmos compatible format.

    - ``drift.csv`` contains drift list
    - ``test_data.pt`` contains cropped images at target sites
    - ``test_target.csv`` contains target site positions
    - ``offset.pt`` contains cropped images of offset sites
    - ``control_data.pt`` contains cropped images at dark sites
    - ``control_target.csv`` contains dark site positions

    Path names to glimpse files are read from the ``options.cfg`` file.
    """

    def get_parser(self, prog_name):
        parser = super(Glimpse, self).get_parser(prog_name)

        parser.add_argument("dataset_path", type=str,
                            help="Path to the dataset folder")

        return parser

    def take_action(self, args):
        read_glimpse(args.dataset_path, D=14)
