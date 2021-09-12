# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def CmdConfig(args):
    import configparser

    config = configparser.ConfigParser(allow_no_value=True)
    config["glimpse"] = {}
    config["glimpse"]["title"] = "title_name"
    config["glimpse"]["dir"] = "/path/to/glimpse/folder"
    config["glimpse"]["ontarget_aoiinfo"] = "/path/to/ontarget_aoiinfo_file"
    config["glimpse"]["offtarget_aoiinfo"] = "/path/to/offtarget_aoiinfo_file"
    config["glimpse"]["driftlist"] = "/path/to/driftlist_file"
    config["glimpse"]["frame_start"] = None
    config["glimpse"]["frame_end"] = None
    config["glimpse"]["ontarget_labels"] = None
    config["glimpse"]["offtarget_labels"] = None

    cfg_file = args.path / "options.cfg"
    with open(cfg_file, "w") as configfile:
        config.write(configfile)


def add_parser(subparsers, parent_parser):
    CONFIG_HELP = "Create options.cfg configuration file with default settings."

    parser = subparsers.add_parser(
        "config",
        parents=[parent_parser],
        description=CONFIG_HELP,
        help=CONFIG_HELP,
    )
    parser.add_argument("path", type=Path, help="Path to the dataset folder")

    parser.set_defaults(func=CmdConfig)
