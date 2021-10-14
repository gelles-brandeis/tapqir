# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0


def CmdLog(args):
    import pydoc

    log_file = args.cd / ".tapqir" / "loginfo"
    with open(log_file, "r") as f:
        pydoc.pager(f.read())


def add_parser(subparsers, parent_parser):
    LOG_HELP = "Show logging info."
    LOG_DESC = """
        Show logging info.
    """

    parser = subparsers.add_parser(
        "log",
        parents=[parent_parser],
        description=LOG_DESC,
        help=LOG_HELP,
    )

    parser.set_defaults(func=CmdLog)
