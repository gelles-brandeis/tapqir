# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def CmdInit(args):
    TAPQIR_PATH = args.path / ".tapqir"
    TAPQIR_PATH.mkdir(exist_ok=True)
    CONFIG_FILE = TAPQIR_PATH / "config"
    CONFIG_FILE.touch(exist_ok=True)


def add_parser(subparsers, parent_parser):
    INIT_HELP = "Initialize Tapqir in the current (or specified) directory."

    parser = subparsers.add_parser(
        "init",
        parents=[parent_parser],
        description=INIT_HELP,
        help=INIT_HELP,
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to the Tapqir folder",
        default=Path.cwd(),
    )

    parser.set_defaults(func=CmdInit)
