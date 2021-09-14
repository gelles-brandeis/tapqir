# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def CmdFit(args):
    from pyroapi import pyro_backend

    from tapqir.models import models

    states = 1
    dtype = "double"
    k_max = args.k
    num_iter = args.it
    batch_size = args.bs
    learning_rate = args.lr
    device = "cuda" if args.cuda else "cpu"
    backend = "funsor" if args.funsor else "pyro"

    # pyro backend
    if backend == "pyro":
        PYRO_BACKEND = "pyro"
    elif backend == "funsor":
        import funsor
        import pyro.contrib.funsor  # noqa: F401

        funsor.set_backend("torch")
        PYRO_BACKEND = "contrib.funsor"
    else:
        raise ValueError("Only pyro and funsor backends are supported.")

    with pyro_backend(PYRO_BACKEND):

        model = models[args.model](states, k_max, device, dtype)
        model.load(args.path)

        model.init(learning_rate, batch_size)
        model.run(num_iter)
        # model.compute_stats()


def add_parser(subparsers, parent_parser):
    FIT_HELP = "Fit the data to the selected model."

    parser = subparsers.add_parser(
        "fit",
        parents=[parent_parser],
        description=FIT_HELP,
        help=FIT_HELP,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Tapqir model to fit the data",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to the Tapqir folder",
        default=Path.cwd(),
    )
    parser.add_argument(
        "-k",
        metavar="<number>",
        default=2,
        type=int,
        help="Maximum number of spots (default: 2)",
    )
    parser.add_argument(
        "-it",
        metavar="<number>",
        type=int,
        help="Number of iterations (default: 0)",
    )
    parser.add_argument(
        "-bs",
        metavar="<number>",
        type=int,
        help="Batch size (default: 5)",
    )
    parser.add_argument(
        "-lr",
        metavar="<number>",
        default=0.005,
        type=float,
        help="Learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--cuda",
        help="Run computations on GPU",
        action="store_true",
    )
    parser.add_argument(
        "--funsor",
        help="Use funsor as backend",
        action="store_true",
    )

    parser.set_defaults(func=CmdFit)
