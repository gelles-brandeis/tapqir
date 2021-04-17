import numpy as np
import pandas as pd
import torch
from pyro.infer import Predictive
from pyroapi import handlers, pyro

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

    # samples
    samples = {}
    samples["gain"] = torch.full((1, 1), params["gain"])
    samples["lamda"] = torch.full((1, 1), params["lamda"])
    samples["proximity"] = torch.full((1, 1), params["proximity"])

    if "pi" in params:
        samples["pi"] = torch.tensor([[1 - params["pi"], params["pi"]]])
        for prefix in ("d", "c"):
            samples[f"{prefix}/background"] = torch.full(
                (1, N, 1), params["background"]
            )
            for k in range(model.K):
                samples[f"{prefix}/width_{k}"] = torch.full((1, N, F), 1.4)
                samples[f"{prefix}/height_{k}"] = torch.full(
                    (1, N, F), params["height"]
                )
    else:
        # kinetic simulations
        samples["init"] = torch.tensor(
            [
                [
                    params["koff"] / (params["kon"] + params["koff"]),
                    params["kon"] / (params["kon"] + params["koff"]),
                ]
            ]
        )
        samples["trans"] = torch.tensor(
            [
                [
                    [1 - params["kon"], params["kon"]],
                    [params["koff"], 1 - params["koff"]],
                ]
            ]
        )
        for f in range(F):
            for prefix in ("d", "c"):
                samples[f"{prefix}/background_{f}"] = torch.full(
                    (1, N, 1), params["background"]
                )
                for k in range(model.K):
                    samples[f"{prefix}/width_{k}_{f}"] = torch.full((1, N, 1), 1.4)
                    samples[f"{prefix}/height_{k}_{f}"] = torch.full(
                        (1, N, 1), params["height"]
                    )

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
    if "pi" in params:
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
