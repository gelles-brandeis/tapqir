# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from cliff.command import Command

from tapqir.imscroll.glimpse_reader import read_glimpse


class Glimpse(Command):
    """
    Read the data from glimpse files and save it in the Cosmos compatible format, ``data.tpqr``.
    Path names to glimpse files are read from the ``options.cfg`` file.
    """

    def get_parser(self, prog_name):
        parser = super(Glimpse, self).get_parser(prog_name)

        parser.add_argument("dataset_path", type=str, help="Path to the dataset folder")

        return parser

    def take_action(self, args):
        read_glimpse(args.dataset_path, P=14)
