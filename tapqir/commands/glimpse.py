# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
from pathlib import Path

from cliff.command import Command

from tapqir.imscroll import read_glimpse


class Glimpse(Command):
    """
    Read the data from glimpse files and save it in the Cosmos compatible format, ``data.tpqr``.
    Path names to glimpse files are read from the ``options.cfg`` file.
    """

    def get_parser(self, prog_name):
        parser = super(Glimpse, self).get_parser(prog_name)

        parser.add_argument(
            "path", default=".", type=str, help="Path to the dataset folder"
        )

        return parser

    def take_action(self, args):
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = Path(args.path) / "options.cfg"
        config.read(cfg_file)
        read_glimpse(
            path=args.path,
            P=14,
            title=config["glimpse"]["title"],
            header_dir=config["glimpse"]["dir"],
            ontarget_aoiinfo=config["glimpse"]["ontarget_aoiinfo"],
            offtarget_aoiinfo=config["glimpse"]["offtarget_aoiinfo"],
            driftlist=config["glimpse"]["driftlist"],
            frame_start=config["glimpse"]["frame_start"],
            frame_end=config["glimpse"]["frame_end"],
            ontarget_labels=config["glimpse"]["ontarget_labels"],
            offtarget_labels=config["glimpse"]["offtarget_labels"],
        )
