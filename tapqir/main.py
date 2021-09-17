# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

from tapqir.cli import parse_args


def main(argv=None):
    """Run tapqir CLI command.

    argv: optional list of commands to parse. sys.argv is used by default.

    """

    args = parse_args(argv)

    # create logger
    logger = logging.getLogger("tapqir")

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if "path" in args and (args.path / ".tapqir").is_dir():
        fh = logging.FileHandler(args.path / ".tapqir" / "log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M %p",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # run command
    if "func" in args:
        args.func(args)


if __name__ == "__main__":
    main()