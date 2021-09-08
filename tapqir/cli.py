import argparse

from .commands import (
        config,
)

COMMANDS = [config]

def get_main_parser():

    # Main parser
    desc = "Tapqir single-molecule data analysis"
    parser = argparse.ArgumentParser(
            prog="tapqir",
            description=desc,
            formatter_class=argparse.RawTextHelpFormatter,
            add_help=False,
    )
    parser.add_argument('--version', action='version', version='%(prog)s 2.0')

    subparsers = parser.add_subparsers()

    for cmd in COMMANDS:
        cmd.add_parser(subparsers, parser)

    return parser

def parse_args(argv=None):
    """Parses CLI arguments."""
    parser = get_main_parser()
    args = parser.parse_args(argv)
    return args
