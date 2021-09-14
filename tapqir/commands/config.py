# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def CmdConfig(args):
    import configparser

    config = configparser.ConfigParser(allow_no_value=True)
    cfg_file = args.path / ".tapqir" / "config"
    config.read(cfg_file)
    section, option = args.name.split(".")
    if section not in config.sections():
        config[section] = {}
    config[section][option] = args.value

    with open(cfg_file, "w") as configfile:
        config.write(configfile)


def add_parser(subparsers, parent_parser):
    CONFIG_HELP = "Set config options."

    parser = subparsers.add_parser(
        "config",
        parents=[parent_parser],
        description=CONFIG_HELP,
        help=CONFIG_HELP,
    )
    parser.add_argument(
        "name",
        type=str,
        help="Option name (command.option)",
    )
    parser.add_argument(
        "value",
        type=str,
        help="Option value",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to the Tapqir folder",
        default=Path.cwd(),
    )

    parser.set_defaults(func=CmdConfig)
