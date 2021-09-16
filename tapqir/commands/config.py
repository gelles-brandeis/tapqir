# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
config
~~~~~~

Description
-----------
"""

from pathlib import Path


def CmdConfig(args):
    import configparser
    import sys

    config = configparser.ConfigParser(allow_no_value=True)
    cfg_file = args.cd / ".tapqir" / "config"
    config.read(cfg_file)

    if args.list:
        config.write(sys.stdout)
        return

    section, option = args.name.split(".")
    if section not in config.sections():
        config[section] = {}
    config[section][option] = args.value

    with open(cfg_file, "w") as configfile:
        config.write(configfile)


def add_parser(subparsers, parent_parser):
    CONFIG_HELP = "Set config options."
    CONFIG_DESC = """
        Set config options.
    """

    parser = subparsers.add_parser(
        "config",
        parents=[parent_parser],
        description=CONFIG_DESC,
        help=CONFIG_HELP,
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Option name (command.option)",
    )
    parser.add_argument(
        "value",
        nargs="?",
        help="Option value",
    )
    parser.add_argument(
        "-l",
        "--list",
        default=False,
        action="store_true",
        help="List all defined config values.",
    )

    parser.set_defaults(func=CmdConfig)
