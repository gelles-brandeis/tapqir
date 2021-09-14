# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def CmdGlimpse(args):
    import configparser

    from tapqir.imscroll import read_glimpse

    # read config file
    config = configparser.ConfigParser(allow_no_value=True)
    cfg_file = args.path / ".tapqir" / "config"
    config.read(cfg_file)
    if "glimpse" not in config.sections():
        config["glimpse"] = {}

    if args.title is not None:
        config["glimpse"]["title"] = args.title
    if args.header_dir is not None:
        config["glimpse"]["dir"] = args.header_dir
    if args.ontarget_aoiinfo is not None:
        config["glimpse"]["ontarget_aoiinfo"] = args.ontarget_aoiinfo
    if args.offtarget_aoiinfo is not None:
        config["glimpse"]["offtarget_aoiinfo"] = args.offtarget_aoiinfo
    if args.driftlist is not None:
        config["glimpse"]["driftlist"] = args.driftlist
    if args.frame_start is not None:
        config["glimpse"]["frame_start"] = args.frame_start
    if args.frame_end is not None:
        config["glimpse"]["frame_end"] = args.frame_end
    if args.ontarget_labels is not None:
        config["glimpse"]["ontarget_labels"] = args.ontarget_labels
    if args.offtarget_labels is not None:
        config["glimpse"]["offtarget_labels"] = args.offtarget_labels

    with open(cfg_file, "w") as configfile:
        config.write(configfile)

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
    parser.add_argument(
        "--title",
        help="Project name",
        metavar="<name>",
    )
    parser.add_argument(
        "--header-dir",
        help="Path to the header/glimpse folder",
        metavar="<path>",
    )
    parser.add_argument(
        "--ontarget-aoiinfo",
        help="Path to the on-target AOI locations file",
        metavar="<path>",
    )
    parser.add_argument(
        "--offtarget-aoiinfo",
        help="Path to the off-target control AOI locations file",
        metavar="<path>",
    )
    parser.add_argument(
        "--driftlist",
        help="Path to the driftlist file",
        metavar="<path>",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        help="First frame to include in the analysis",
        metavar="<number>",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="Last frame to include in the analysis",
        metavar="<number>",
    )
    parser.add_argument(
        "--ontarget-labels",
        help="On-target AOI binding labels",
        metavar="<path>",
    )
    parser.add_argument(
        "--offtarget-labels",
        help="Off-target AOI binding labels",
        metavar="<path>",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to the Tapqir folder",
        default=Path.cwd(),
    )

    parser.set_defaults(func=CmdGlimpse)
