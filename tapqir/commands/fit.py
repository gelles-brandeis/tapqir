# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
fit
~~~

Fit the data to the selected model.

Description
-----------

Fitting
"""


def CmdFit(args):
    import configparser

    from pyroapi import pyro_backend

    from tapqir.models import models

    # read config file
    config = configparser.ConfigParser(allow_no_value=True)
    cfg_file = args.cd / ".tapqir" / "config"
    config.read(cfg_file)
    if "fit" not in config.sections():
        config["fit"] = {}

    states = 1
    dtype = "double"
    k_max = args.k_max if args.k_max is not None else config["fit"].getint("k_max", 2)
    batch_size = args.bs if args.bs is not None else config["fit"].getint("bs", 0)
    learning_rate = (
        args.lr if args.lr is not None else config["fit"].getfloat("lr", 0.005)
    )
    num_iter = (
        args.num_iter
        if args.num_iter is not None
        else config["fit"].getint("num_iter", 0)
    )
    marginal = args.marginal
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

        model = models[args.model](states, k_max, device, dtype, marginal)
        model.load(args.cd)

        model.init(learning_rate, batch_size)
        model.run(num_iter)


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
        "--marginal",
        help="Marginalize out theta in the model (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--cuda",
        help="Run computations on GPU",
        action="store_true",
    )
    parser.add_argument(
        "--bs",
        metavar="<number>",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        metavar="<number>",
        type=float,
        help="Learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--num-iter",
        metavar="<number>",
        type=int,
        help="Number of iterations (default: 0)",
    )
    parser.add_argument(
        "--k-max",
        metavar="<number>",
        type=int,
        help="Maximum number of spots (default: 2)",
    )
    parser.add_argument(
        "--funsor",
        help="Use funsor as backend",
        action="store_true",
    )

    parser.set_defaults(func=CmdFit)
