# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from pyroapi import distributions as dist
from pyroapi import pyro, pyro_backend

from tapqir.models import Cosmos, CosmosMarginal
from tapqir.utils.dataset import save
from tapqir.utils.simulate import simulate


def main(args):
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"
    pyro.set_rng_seed(args.seed)
    params = {}
    params["width"] = 1.4
    params["gain"] = args.gain if args.gain is not None else random.uniform(1, 20)
    params["pi"] = args.pi if args.pi is not None else random.betavariate(1, 9)
    params["lamda"] = args.lamda if args.lamda is not None else random.uniform(0, 1)
    params["proximity"] = (
        args.proximity if args.proximity is not None else random.uniform(0.2, 0.6)
    )
    params["offset"] = 90
    params["height"] = args.height
    params["background"] = 150

    # calculate snr

    model = Cosmos(1, 2, device, args.dtype)
    if args.path is not None:
        data_path = Path(args.path)
        try:
            model.load(data_path)
        except FileNotFoundError:
            simulate(
                model,
                args.N,
                args.F,
                args.P,
                seed=args.seed,
                params=params,
            )
            # save data
            if not data_path.is_dir():
                data_path.mkdir()
            save(model.data, data_path)
            # calculate snr
            rv = dist.MultivariateNormal(
                torch.tensor([(args.P - 1) / 2, (args.P - 1) / 2]),
                scale_tril=torch.eye(2) * params["width"],
            )

            D_range = torch.arange(args.P, dtype=model.dtype)
            i_pixel, j_pixel = torch.meshgrid(D_range, D_range)
            ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)
            weights = rv.log_prob(ij_pixel).exp()
            signal = (weights ** 2 * params["height"]).sum()
            noise = math.sqrt((params["background"] * params["gain"]))
            params["SNR"] = float(signal / noise)
            params["N"] = args.N
            params["F"] = args.F
            params["Nc"] = args.N
            params["Fc"] = args.F
            params["P"] = args.P

            pd.Series(params).to_csv(Path(data_path) / "simulated_params.csv")
            pyro.clear_param_store()
    else:
        simulate(model, args.N, args.F, args.P, seed=args.seed, params=params)
        pyro.clear_param_store()

    marginal = CosmosMarginal(device=device, dtype=args.dtype)
    marginal.load(args.path)
    marginal.init(args.lr, args.bs, args.jit)
    marginal.run(args.it)

    model.init(args.lr, args.bs, args.jit)
    model.run(args.it)
    model.compute_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Height Simulations")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gain", type=float)  # default 7.0
    parser.add_argument("--pi", type=float)  # default 0.15
    parser.add_argument("--lamda", type=float)  # default 0.15
    parser.add_argument("--proximity", type=float)  # default 0.2
    parser.add_argument("--height", default=3000, type=int)  # default 3000
    parser.add_argument("-N", default=5, type=int)
    parser.add_argument("-F", default=500, type=int)
    parser.add_argument("-P", default=14, type=int)
    parser.add_argument("-it", default=70000, type=int)
    parser.add_argument("-bs", default=0, type=int)
    parser.add_argument("-lr", default=0.005, type=float)
    parser.add_argument("--path", type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dtype", default="float", type=str)
    parser.add_argument("--funsor", action="store_true")
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()

    if args.funsor:
        import funsor

        funsor.set_backend("torch")
        import pyro.contrib.funsor

        PYRO_BACKEND = "contrib.funsor"
    else:
        PYRO_BACKEND = "pyro"

    with pyro_backend(PYRO_BACKEND):
        main(args)
