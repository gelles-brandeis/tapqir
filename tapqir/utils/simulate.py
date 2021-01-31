import numpy as np
import pandas as pd
import torch
from pyro.infer import Predictive
from pyroapi import handlers, pyro
from torch.distributions import constraints

from tapqir.models import GaussianSpot
from tapqir.utils.dataset import CosmosDataset


def simulate(model, N, F, D=14, seed=0, cuda=True, params=dict()):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        device = torch.device("cpu")

    # parameters and samples
    pyro.param("gain", torch.tensor(params["gain"]), constraint=constraints.positive)
    pyro.param("width_mean", torch.tensor(1.5), constraint=constraints.positive)
    pyro.param("width_size", torch.tensor(2.0), constraint=constraints.positive)
    if "probs_z" in params:
        pyro.param(
            "probs_z",
            torch.tensor([1 - params["probs_z"], params["probs_z"]]),
            constraint=constraints.simplex,
        )
    else:
        pyro.param(
            "init_z",
            torch.tensor(
                [
                    params["koff"] / (params["kon"] + params["koff"]),
                    params["kon"] / (params["kon"] + params["koff"]),
                ]
            ),
            constraint=constraints.simplex,
        )
        pyro.param(
            "trans_z",
            torch.tensor(
                [
                    [1 - params["kon"], params["kon"]],
                    [params["koff"], 1 - params["koff"]],
                ]
            ),
            constraint=constraints.simplex,
        )

    pyro.param(
        "rate_j", torch.tensor(params["rate_j"]), constraint=constraints.positive
    )
    pyro.param(
        "proximity",
        torch.tensor([params["proximity"]]),
        constraint=constraints.positive,
    )

    if "probs_z" in params:
        samples = {
            "d/background": torch.full((1, N, 1), params["background"]),
            "d/width_0": torch.full((1, N, F), 1.4),
            "d/width_1": torch.full((1, N, F), 1.4),
            "d/height_0": torch.full((1, N, F), params["height"]),
            "d/height_1": torch.full((1, N, F), params["height"]),
            "c/background": torch.full((1, N, 1), params["background"]),
            "c/width_0": torch.full((1, N, F), 1.4),
            "c/width_1": torch.full((1, N, F), 1.4),
            "c/height_0": torch.full((1, N, F), params["height"]),
            "c/height_1": torch.full((1, N, F), params["height"]),
        }
    else:
        # kinetic simulations
        samples = {}
        for f in range(F):
            samples[f"d/background_{f}"] = torch.full((1, N, 1), params["background"])
            samples[f"d/width_0_{f}"] = torch.full((1, N, 1), 1.4)
            samples[f"d/width_1_{f}"] = torch.full((1, N, 1), 1.4)
            samples[f"d/height_0_{f}"] = torch.full((1, N, 1), params["height"])
            samples[f"d/height_1_{f}"] = torch.full((1, N, 1), params["height"])
            samples[f"c/background_{f}"] = torch.full((1, N, 1), params["background"])
            samples[f"c/width_0_{f}"] = torch.full((1, N, 1), 1.4)
            samples[f"c/width_1_{f}"] = torch.full((1, N, 1), 1.4)
            samples[f"c/height_0_{f}"] = torch.full((1, N, 1), params["height"])
            samples[f"c/height_1_{f}"] = torch.full((1, N, 1), params["height"])

    offset = torch.full((3,), params["offset"])
    target = pd.DataFrame(
        data={"frame": np.zeros(N), "x": 6.5, "y": 6.5}, index=np.arange(N)
    )
    target.index.name = "aoi"
    drift = pd.DataFrame(data={"dx": 0.0, "dy": 0}, index=np.arange(F))
    drift.index.name = "frame"
    model.data = CosmosDataset(
        torch.full((N, F, D, D), params["background"] + params["offset"]),
        target,
        drift,
        dtype="test",
        device=device,
        offset=offset,
    )
    model.control = CosmosDataset(
        torch.zeros(N, F, D, D), target, drift, dtype="control", device=device
    )
    model.data_loc = GaussianSpot(model.data.target, model.data.drift, model.data.D)
    model.control_loc = GaussianSpot(
        model.control.target, model.control.drift, model.control.D
    )

    # sample
    predictive = Predictive(
        handlers.uncondition(model.model), posterior_samples=samples, num_samples=1
    )
    samples = predictive()
    model.data.labels = np.zeros(
        (N, F), dtype=[("aoi", int), ("frame", int), ("z", bool)]
    )
    model.data.labels["aoi"] = np.arange(N).reshape(-1, 1)
    model.data.labels["frame"] = np.arange(F)
    if "probs_z" in params:
        model.data.data = samples["d/data"][0].data.floor()
        model.control.data = samples["c/data"][0].data.floor()
        model.data.labels["z"] = samples["d/theta"][0].cpu() > 0
    else:
        # kinetic simulations
        for f in range(F):
            model.data.data[:, f : f + 1] = samples[f"d/data_{f}"][0].data.floor()
            model.control.data[:, f : f + 1] = samples[f"c/data_{f}"][0].data.floor()
            model.data.labels["z"][:, f : f + 1] = samples[f"d/theta_{f}"][0].cpu() > 0

    return model
