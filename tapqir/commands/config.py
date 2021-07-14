# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
from pathlib import Path

from cliff.command import Command


class Config(Command):
    """
    Create ``options.cfg`` configuration file with default settings.
    """

    def get_parser(self, prog_name):
        parser = super(Config, self).get_parser(prog_name)

        parser.add_argument("path", type=str, help="Path to the dataset folder")

        return parser

    def take_action(self, args):
        config = configparser.ConfigParser(allow_no_value=True)
        config["glimpse"] = {}
        config["glimpse"]["dir"] = "/path/to/glimpse/folder"
        config["glimpse"]["ontarget_aoiinfo"] = "/path/to/ontarget_aoiinfo_file"
        config["glimpse"]["offtarget_aoiinfo"] = "/path/to/offtarget_aoiinfo_file"
        config["glimpse"]["driftlist"] = "/path/to/driftlist_file"
        config["glimpse"]["frame_start"] = None
        config["glimpse"]["frame_end"] = None
        config["glimpse"]["ontarget_labels"] = None
        config["glimpse"]["offtarget_labels"] = None

        config["fit"] = {}
        config["fit"]["num_states"] = "1"
        config["fit"]["k_max"] = "2"
        config["fit"]["num_iter"] = "70000"
        config["fit"]["batch_size"] = "0"
        config["fit"]["learning_rate"] = "0.005"
        config["fit"]["device"] = "cuda"
        config["fit"]["dtype"] = "double"
        config["fit"]["jit"] = "False"
        config["fit"]["backend"] = "pyro"

        cfg_file = Path(args.path) / "options.cfg"
        with open(cfg_file, "w") as configfile:
            config.write(configfile)
