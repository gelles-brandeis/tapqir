# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from tapqir import __version__ as tapqir_version

from .commands import config, fit, glimpse, init, log, show, stats

COMMANDS = [
    init,
    glimpse,
    fit,
    stats,
    show,
    config,
    log,
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
    )
    parser.add_argument("-v", "--version", action="version", version=tapqir_version)
    parser.add_argument(
        "--cd",
        type=Path,
        help="Change working directory (default: current working directory)",
        default=Path.cwd(),
        metavar="<path>",
    )

    # subparsers = parser.add_subparsers()
    subparsers = parser.add_subparsers(
        title="Available sub-commands",
        metavar="<command>",
        dest="cmd",
        help="Use `tapqir <command> -h` for command-specific help.",
    )

    for cmd in COMMANDS:
        cmd.add_parser(subparsers, parent_parser)

    return parser


def parse_args(argv=None):
    """Parses CLI arguments."""
    parser = get_main_parser()
    args = parser.parse_args()
    return args
