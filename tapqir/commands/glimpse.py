# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0


def CmdGlimpse(args):
    import configparser

    from tapqir.imscroll import read_glimpse

    # read config file
    config = configparser.ConfigParser(allow_no_value=True)
    cfg_file = args.cd / ".tapqir" / "config"
    config.read(cfg_file)
    if "glimpse" not in config.sections():
        config["glimpse"] = {}

    kwargs = {}
    kwargs["title"] = (
        args.title if args.title is not None else config["glimpse"].get("title", None)
    )
    kwargs["aoi_size"] = (
        args.aoi_size
        if args.aoi_size is not None
        else config["glimpse"].get("aoi_size", 14)
    )
    kwargs["header_dir"] = (
        args.header_dir
        if args.header_dir is not None
        else config["glimpse"].get("header_dir", None)
    )
    kwargs["ontarget_aoiinfo"] = (
        args.ontarget_aoiinfo
        if args.ontarget_aoiinfo is not None
        else config["glimpse"].get("ontarget_aoiinfo", None)
    )
    kwargs["offtarget_aoiinfo"] = (
        args.offtarget_aoiinfo
        if args.offtarget_aoiinfo is not None
        else config["glimpse"].get("offtarget_aoiinfo", None)
    )
    kwargs["driftlist"] = (
        args.driftlist
        if args.driftlist is not None
        else config["glimpse"].get("driftlist", None)
    )
    kwargs["frame_start"] = (
        args.frame_start
        if args.frame_start is not None
        else config["glimpse"].get("frame_start", None)
    )
    kwargs["frame_end"] = (
        args.frame_end
        if args.frame_end is not None
        else config["glimpse"].get("frame_end", None)
    )
    kwargs["ontarget_labels"] = (
        args.ontarget_labels
        if args.ontarget_labels is not None
        else config["glimpse"].get("ontarget_labels", None)
    )
    kwargs["offtarget_labels"] = (
        args.offtarget_labels
        if args.offtarget_labels is not None
        else config["glimpse"].get("offtarget_labels", None)
    )

    read_glimpse(
        path=args.cd,
        **kwargs,
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
        help="Project/experiment name",
        metavar="<name>",
    )
    parser.add_argument(
        "--aoi-size",
        help="AOI image size - number of pixels along the axis (default: 14)",
        metavar="<number>",
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
        help="Path to the off-target control AOI locations file (optional)",
        metavar="<path>",
    )
    parser.add_argument(
        "--driftlist",
        help="Path to the driftlist file",
        metavar="<path>",
    )
    parser.add_argument(
        "--frame-start",
        help="First frame to include in the analysis (optional)",
        metavar="<number>",
    )
    parser.add_argument(
        "--frame-end",
        help="Last frame to include in the analysis (optional)",
        metavar="<number>",
    )
    parser.add_argument(
        "--ontarget-labels",
        help="On-target AOI binding labels (optional)",
        metavar="<path>",
    )
    parser.add_argument(
        "--offtarget-labels",
        help="Off-target AOI binding labels (optional)",
        metavar="<path>",
    )

    parser.set_defaults(func=CmdGlimpse)
