# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def CmdGlimpse(args):
    from tapqir.imscroll import read_glimpse

    read_glimpse(
        path=args.path,
        P=14,
    )


def add_parser(subparsers, parent_parser):
    GLIMPSE_HELP = "Extract AOIs from raw images."

    parser = subparsers.add_parser(
        "glimpse",
        parents=[parent_parser],
        description=GLIMPSE_HELP,
        help=GLIMPSE_HELP,
    )
    parser.add_argument("path", type=Path, help="Path to the dataset folder")

    parser.set_defaults(func=CmdGlimpse)
