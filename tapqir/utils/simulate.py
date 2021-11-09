# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from pyro.infer import Predictive
from pyroapi import handlers, pyro

from tapqir.utils.dataset import CosmosDataset


def simulate(
    model: str,
    N: int,
    F: int,
    C: int = 1,
    P: int = 14,
    seed: int = 0,
    params: dict = dict(),
) -> CosmosDataset:
    """
    Simulate a new dataset.

    :param model: Tapqir model.
    :param N: Number of total AOIs. Half will be on-target and half off-target.
    :param F: Number of frames.
    :param C: Number of color channels.
    :param P: Number of pixels alongs the axis.
    :param seed: Rng seed.
    :param params: A dictionary of fixed parameter values.
    :return: A new simulated data set.
    """

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    # samples
    samples = {}
    samples["gain"] = torch.full((1, 1), params["gain"])
    samples["lamda"] = torch.full((1, 1), params["lamda"])
    samples["proximity"] = torch.full((1, 1), params["proximity"])

    if "pi" in params:
        samples["pi"] = torch.tensor([[1 - params["pi"], params["pi"]]])
        samples["background"] = torch.full((1, N, 1), params["background"])
        for k in range(model.K):
            samples[f"width_{k}"] = torch.full((1, N, F), params["width"])
            samples[f"height_{k}"] = torch.full((1, N, F), params["height"])
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
            samples[f"background_{f}"] = torch.full((1, N, 1), params["background"])
            for k in range(model.K):
                samples[f"width_{k}_{f}"] = torch.full((1, N, 1), params["width"])
                samples[f"height_{k}_{f}"] = torch.full((1, N, 1), params["height"])

    offset = torch.full((3,), params["offset"])
    target_locs = torch.full((N, F, 1, 2), (P - 1) / 2)
    is_ontarget = torch.zeros((N,), dtype=torch.bool)
    is_ontarget[: N // 2] = True
    # placeholder dataset
    model.data = CosmosDataset(
        torch.full((N, F, C, P, P), params["background"] + params["offset"]),
        target_locs,
        is_ontarget,
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
    data = torch.zeros(N, F, C, P, P)
    labels = np.zeros((N // 2, F, 1), dtype=[("aoi", int), ("frame", int), ("z", bool)])
    labels["aoi"] = np.arange(N // 2).reshape(-1, 1, 1)
    labels["frame"] = np.arange(F).reshape(-1, 1)
    if "pi" in params:
        data[:, :, 0] = samples["data"][0].data.floor()
        labels["z"][:, :, 0] = samples["theta"][0][: N // 2].cpu() > 0
    else:
        # kinetic simulations
        for f in range(F):
            data[:, f : f + 1, 0] = samples[f"data_{f}"][0].data.floor()
            labels["z"][:, f : f + 1, 0] = samples[f"theta_{f}"][0][: N // 2].cpu() > 0

    return CosmosDataset(
        data.cpu(),
        target_locs.cpu(),
        is_ontarget.cpu(),
        labels,
        offset_samples=offset,
        offset_weights=torch.ones(3) / 3,
        device=model.device,
    )
