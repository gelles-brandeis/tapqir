# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from tapqir.cli import parse_args


def main(argv=None):
    """Run tapqir CLI command.

    argv: optional list of commands to parse. sys.argv is used by default.

    """
    parse_args(argv)


if __name__ == "__main__":
    main()
