# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse

from tapqir import __version__ as tapqir_version

from .commands import config, fit, glimpse, init, save, show

COMMANDS = [
    config,
    fit,
    glimpse,
    init,
    save,
    show,
]


def get_main_parser():
    parent_parser = argparse.ArgumentParser(add_help=False)

    # Main parser
    desc = "Bayesian analysis of single-molecule image data"
    parser = argparse.ArgumentParser(
        prog="tapqir",
        description=desc,
        parents=[parent_parser],
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    parser.add_argument("-v", "--version", action="version", version=tapqir_version)

    # subparsers = parser.add_subparsers()
    subparsers = parser.add_subparsers(
        title="Available Commands",
        metavar="COMMAND",
        dest="cmd",
        help="Use `tapqir COMMAND --help` for command-specific help.",
    )

    for cmd in COMMANDS:
        cmd.add_parser(subparsers, parent_parser)

    return parser


def parse_args(argv=None):
    """Parses CLI arguments."""
    parser = get_main_parser()
    args = parser.parse_args()
    args.func(args)
    return args
