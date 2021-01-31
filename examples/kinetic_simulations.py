import argparse
from pathlib import Path

import pandas as pd
import torch
from pyroapi import pyro, pyro_backend

from tapqir.models import HMM, Cosmos
from tapqir.utils.simulate import simulate


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = "cuda"
    else:
        device = "cpu"
    pyro.set_rng_seed(args.seed)
    params = {}
    params["gain"] = 7.0
    params["kon"] = args.kon
    params["koff"] = 0.2
    params["rate_j"] = args.ratej
    params["proximity"] = 0.2
    params["offset"] = 90.0
    params["height"] = 3000
    params["background"] = 150

    model = Cosmos(1, 2)
    data_path = args.path or Path("data") / f"kon{args.kon:.2e}ratej{args.ratej:.2e}"
    try:
        model.load(data_path, True, device)
    except FileNotFoundError:
        hmm = HMM(1, 2)
        hmm.vectorized = False
        simulate(
            hmm, args.N, args.F, args.D, seed=args.seed, cuda=args.cuda, params=params
        )
        # save data
        hmm.data.save(data_path)
        hmm.control.save(data_path)
        pd.Series(params).to_csv(Path(data_path) / "simulated_params.csv")
        pyro.clear_param_store()
        model.load(data_path, True, device)

    model.settings(args.lr, args.bs)
    model.run(args.it, args.infer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Simulations")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--kon", default=1e-3, type=float)
    parser.add_argument("--ratej", default=0.15, type=float)
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
