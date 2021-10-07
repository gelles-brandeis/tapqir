# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from pyro.infer import Predictive
from pyroapi import handlers, pyro

from tapqir.utils.dataset import CosmosDataset


def simulate(model, N, F, P=14, seed=0, params=dict()):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

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
            samples[f"d/background_{f}"] = torch.full((1, N, 1), params["background"])
            for k in range(model.K):
                samples[f"d/width_{k}_{f}"] = torch.full((1, N, 1), params["width"])
                samples[f"d/height_{k}_{f}"] = torch.full((1, N, 1), params["height"])
        samples["c/background"] = torch.full((1, N, 1), params["background"])
        for k in range(model.K):
            samples[f"c/width_{k}"] = torch.full((1, N, F), params["width"])
            samples[f"c/height_{k}"] = torch.full((1, N, F), params["height"])

    offset = torch.full((3,), params["offset"])
    target_locs = torch.full((N, F, 2), (P - 1) / 2)
    model.data = CosmosDataset(
        torch.full((N, F, P, P), params["background"] + params["offset"]),
        target_locs,
        None,
        torch.full((N, F, P, P), params["background"] + params["offset"]),
        target_locs,
        None,
        offset_samples=offset,
        offset_weights=torch.ones(3) / 3,
        device=model.device,
    )

    # sample
    predictive = Predictive(
        handlers.uncondition(model.model), posterior_samples=samples, num_samples=1
    )
    samples = predictive()
    data = torch.zeros(N, F, P, P)
    control = torch.zeros(N, F, P, P)
    labels = np.zeros((N, F), dtype=[("aoi", int), ("frame", int), ("z", bool)])
    labels["aoi"] = np.arange(N).reshape(-1, 1)
    labels["frame"] = np.arange(F)
    if "pi" in params:
        data = samples["d/data"][0].data.floor()
        control = samples["c/data"][0].data.floor()
        labels["z"] = samples["d/theta"][0].cpu() > 0
    else:
        # kinetic simulations
        for f in range(F):
            data[:, f : f + 1] = samples[f"d/data_{f}"][0].data.floor()
            labels["z"][:, f : f + 1] = samples[f"d/theta_{f}"][0].cpu() > 0
        control = samples["c/data"][0].data.floor()
    model.data = CosmosDataset(
        data.cpu(),
        target_locs.cpu(),
        labels,
        control.cpu(),
        target_locs.cpu(),
        None,
        offset_samples=offset,
        offset_weights=torch.ones(3) / 3,
        device=model.device,
    )

    return model
