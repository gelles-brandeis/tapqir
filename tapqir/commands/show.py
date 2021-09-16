# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later


def CmdShow(args):
    import sys

    from pyroapi import pyro_backend
    from PySide2.QtWidgets import QApplication

    from tapqir.commands.qtgui import MainWindow
    from tapqir.models import models

    backend = "funsor" if args.funsor else "pyro"
    # pyro backend
    if backend == "pyro":
        PYRO_BACKEND = "pyro"
    elif backend == "funsor":
        PYRO_BACKEND = "contrib.funsor"
    else:
        raise ValueError("Only pyro and funsor backends are supported.")

    with pyro_backend(PYRO_BACKEND):
        model = models[args.model](1, 2, "cpu", "float")
        app = QApplication(sys.argv)
        MainWindow(model, args.cd)
        sys.exit(app.exec_())


def add_parser(subparsers, parent_parser):
    SHOW_HELP = "Show fitting results."

    parser = subparsers.add_parser(
        "show",
        parents=[parent_parser],
        description=SHOW_HELP,
        help=SHOW_HELP,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Tapqir model to fit the data",
    )
    parser.add_argument(
        "--funsor",
        help="Use funsor as backend",
        action="store_true",
    )

    parser.set_defaults(func=CmdShow)
