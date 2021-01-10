import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from pyroapi import pyro, pyro_backend

from tapqir.utils.simulate import simulate


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = "cuda"
    else:
        device = "cpu"
    pyro.set_rng_seed(args.seed)
    params = {}
    params["gain"] = random.uniform(1, 20)
    params["probs_z"] = random.betavariate(1, 9)
    params["rate_j"] = random.uniform(0, 1)
    params["proximity"] = random.uniform(0.2, 0.6)
    params["offset"] = 90.0
    params["height"] = 3000
    params["background"] = 150

    model = simulate(args.N, args.F, args.D, cuda=args.cuda, params=params)

    # save data
    model.data_path = args.path or Path("data") / "seed{}".format(args.seed)
    model.data.save(model.data_path)
    model.control.save(model.data_path)
    pd.Series(params).to_csv(Path(model.data_path) / "simulated_params.csv")

    model.load(model.data_path, True, device)
    model.settings(args.lr, args.bs)
    model.run(args.it, args.infer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomized Simulations")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-N", default=5, type=int)
    parser.add_argument("-F", default=500, type=int)
    parser.add_argument("-D", default=14, type=int)
    parser.add_argument("-it", default=100, type=int)
    parser.add_argument("-infer", default=100, type=int)
    parser.add_argument("-bs", default=0, type=int)
    parser.add_argument("-lr", default=0.005, type=float)
    parser.add_argument("--path", type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--funsor", action="store_true")
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
