# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import colorama


def format_link(link):
    return "<{blue}{link}{nc}>".format(
        blue=colorama.Fore.CYAN, link=link, nc=colorama.Fore.RESET
    )


def CmdInit(args):
    TAPQIR_PATH = args.cd / ".tapqir"
    TAPQIR_PATH.mkdir(exist_ok=True)
    CONFIG_FILE = TAPQIR_PATH / "config"
    CONFIG_FILE.touch(exist_ok=True)

    print(
        (
            "Initialized Tapqir analysis folder.\n"
            "\n"
            "{yellow}Tapqir is a Bayesian program for single-molecule data analysis.{nc}\n"
            "{yellow}---------------------------------------------------------------{nc}\n"
            f"- Checkout the documentation: {format_link('https://tapqir.readthedocs.io/')}\n"
            f"- Get help on our forum: {format_link('https://github.com/gelles-brandeis/tapqir/discussions')}\n"
            f"- Star us on GitHub: {format_link('https://github.com/gelles-brandeis/tapqir')}\n"
        ).format(yellow=colorama.Fore.YELLOW, nc=colorama.Fore.RESET)
    )


def add_parser(subparsers, parent_parser):
    INIT_HELP = "Initialize Tapqir in the current (or specified) directory."
    INIT_DESC = """
        Initialize Tapqir workspace in the current (or specified) directory.

        Initializing a Tapqir workspace will create ``.tapqir`` directory containing
        ``config`` file, ``log`` file, and files created by commands such as ``tapqir fit``.
    """

    parser = subparsers.add_parser(
        "init",
        parents=[parent_parser],
        description=INIT_DESC,
        help=INIT_HELP,
    )

    parser.set_defaults(func=CmdInit)
