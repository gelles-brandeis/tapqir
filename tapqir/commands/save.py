# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
stats
~~~~~

Fit the data to the selected model.

Description
-----------
"""

from pathlib import Path


def CmdStats(args):
    from pyroapi import pyro_backend

    from tapqir.models import models
    from tapqir.utils.stats import save_stats

    backend = "funsor" if args.funsor else "pyro"
    dtype = "double"

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

        model = models[args.model](1, 2, "cpu", dtype)
        model.load(args.cd)
        model.load_checkpoint(param_only=True)

        save_stats(model, args.cd, save_matlab=args.matlab)


def add_parser(subparsers, parent_parser):
    STATS_HELP = "Compute credible intervals for model parameters."

    parser = subparsers.add_parser(
        "stats",
        parents=[parent_parser],
        description=STATS_HELP,
        help=STATS_HELP,
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
    parser.add_argument(
        "--matlab",
        help="Save parameters in matlab format (default: False)",
        action="store_true",
    )
    parser.set_defaults(func=CmdStats)
