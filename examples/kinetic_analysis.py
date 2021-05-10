import argparse
from pathlib import Path

import pandas as pd
import torch
from pyro.ops.stats import pi
from pyroapi import distributions as dist

from tapqir.models import Cosmos
from tapqir.utils.imscroll import (
    bound_dwell_times,
    count_intervals,
    unbound_dwell_times,
)
from tapqir.utils.stats import save_stats


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = "cuda"
    else:
        device = "cpu"

    model = Cosmos(1, 2)
    model.load(args.data_path, False, device)
    model.load_parameters(args.param_path)

    save_stats(model, args.param_path)

    kon = torch.zeros(args.it)
    koff = torch.zeros(args.it)
    for i in range(args.it):
        z_trace = dist.Bernoulli(model.z_marginal).sample()
        intervals = count_intervals(z_trace)
        kon[i] = 1 / unbound_dwell_times(intervals).mean()
        koff[i] = 1 / bound_dwell_times(intervals).mean()
    kon_ll, kon_ul = pi(kon, 0.68)
    kon_mean = kon.mean()
    koff_ll, koff_ul = pi(koff, 0.68)
    koff_mean = koff.mean()

    df = pd.read_csv(
        Path(args.param_path) / "statistics.csv", squeeze=True, index_col=0
    )

    # rate constants
    df["kon_mean"] = kon_mean.item()
    df["kon_ll"] = kon_ll.item()
    df["kon_ul"] = kon_ul.item()
    df["koff_mean"] = koff_mean.item()
    df["koff_ll"] = koff_ll.item()
    df["koff_ul"] = koff_ul.item()

    # equilibrium constant
    df["keq"] = df["pi_mean"] / (1 - df["pi_mean"])
    df.to_csv(Path(args.param_path) / "statistics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Simulations")
    parser.add_argument("-it", default=100, type=int)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--param-path", type=str)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    main(args)
