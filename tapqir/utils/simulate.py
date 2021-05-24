# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
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
                samples[f"{prefix}/width_{k}"] = torch.full((1, N, F), params["width"])
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
            samples["d/background_{f}"] = torch.full((1, N, 1), params["background"])
            for k in range(model.K):
                samples["d/width_{k}_{f}"] = torch.full((1, N, 1), params["width"])
                samples["d/height_{k}_{f}"] = torch.full((1, N, 1), params["height"])
        samples["c/background"] = torch.full((1, N, 1), params["background"])
        for k in range(model.K):
            samples["c/width_{k}"] = torch.full((1, N, F), params["width"])
            samples["c/height_{k}"] = torch.full((1, N, F), params["height"])

    offset = torch.full((3,), params["offset"])
    target_locs = torch.full((N, F, 2), (D - 1) / 2)
    model.data = CosmosDataset(
        torch.full((N, F, D, D), params["background"] + params["offset"]),
        target_locs,
        dtype="test",
        device=device,
        offset=offset,
    )
    model.control = CosmosDataset(
        torch.zeros(N, F, D, D), target_locs, dtype="control", device=device
    )
    model.data_loc = GaussianSpot(model.data.target_locs, model.data.D)
    model.control_loc = GaussianSpot(model.control.target_locs, model.control.D)

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
            model.data.labels["z"][:, f : f + 1] = samples[f"d/theta_{f}"][0].cpu() > 0
        model.control.data = samples["c/data"][0].data.floor()

    return model
