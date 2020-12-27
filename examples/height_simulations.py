import argparse
from pathlib import Path

import pandas as pd
import pyro
import torch

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
    params["probs_z"] = 0.15
    params["rate_j"] = 0.15
    params["proximity"] = 0.2
    params["offset"] = 90.0
    params["height"] = args.height
    params["background"] = 150

    model = simulate(args.N, args.F, args.D, cuda=args.cuda, params=params)

    # save data
    model.data_path = args.path or "height{}".format(args.height)
    model.data.save(model.data_path)
    model.control.save(model.data_path)
    pd.Series(params).to_csv(Path(model.data_path) / "simulated_params.csv")

    model.load(model.data_path, True, device)
    model.settings(args.lr, args.bs)
    model.run(args.it, args.infer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Height Simulations")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--height", default=3000, type=int)
    parser.add_argument("-N", default=5, type=int)
    parser.add_argument("-F", default=500, type=int)
    parser.add_argument("-D", default=14, type=int)
    parser.add_argument("-it", default=100, type=int)
    parser.add_argument("-infer", default=100, type=int)
    parser.add_argument("-bs", default=0, type=int)
    parser.add_argument("-lr", default=0.005, type=float)
    parser.add_argument("--path", type=str)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)