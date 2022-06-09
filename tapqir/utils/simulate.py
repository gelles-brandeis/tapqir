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
    Q = C

    # samples
    samples = {}
    samples["gain"] = torch.full((1, 1), params["gain"])
    # samples["gain"] = torch.as_tensor(params["gain"])
    samples["lamda"] = torch.full((1, Q), params["lamda"])
    samples["proximity"] = torch.full((1, 1), params["proximity"])

    if "alpha" in params:
        samples["alpha"] = torch.as_tensor(params["alpha"]).expand([1, Q, C])
        samples["pi"] = torch.tensor([[1 - params["pi"], params["pi"]]]).expand(1, Q, 2)
        samples["background"] = torch.full((1, N, 1, C), params["background"])
        for q in range(Q):
            for k in range(model.K):
                samples[f"width_k{k}_q{q}"] = torch.full((1, N, F), params["width"])
                samples[f"height_k{k}_q{q}"] = torch.full((1, N, F), params["height"])
    elif "pi" in params:
        samples["pi"] = torch.tensor([[1 - params["pi"], params["pi"]]]).expand(1, Q, 2)
        samples["background"] = torch.full((1, N, 1, C), params["background"])
        for k in range(model.K):
            samples[f"width_k{k}"] = torch.full((1, N, F, Q), params["width"])
            samples[f"height_k{k}"] = torch.full((1, N, F, Q), params["height"])
    elif ("init" in params) and ("trans" in params):
        samples["init"] = params["init"]
        samples["trans"] = params["trans"]
        for f in range(F):
            samples[f"background_f{f}"] = torch.full((1, N, 1, C), params["background"])
            for k in range(model.K):
                samples[f"width_k{k}_f{f}"] = torch.full((1, N, 1, Q), params["width"])
    elif ("kon" in params) and ("koff" in params):
        # kinetic simulations
        samples["init"] = torch.tensor(
            [
                [
                    params["koff"] / (params["kon"] + params["koff"]),
                    params["kon"] / (params["kon"] + params["koff"]),
                ]
            ]
        ).expand(1, Q, 2)
        samples["trans"] = torch.tensor(
            [
                [
                    [1 - params["kon"], params["kon"]],
                    [params["koff"], 1 - params["koff"]],
                ]
            ]
        ).expand(1, Q, 2, 2)
        for f in range(F):
            samples[f"background_f{f}"] = torch.full((1, N, 1, C), params["background"])
            for k in range(model.K):
                samples[f"width_k{k}_f{f}"] = torch.full((1, N, 1, Q), params["width"])
                samples[f"height_k{k}_f{f}"] = torch.full(
                    (1, N, 1, Q), params["height"]
                )

    offset = torch.full((3,), params["offset"])
    target_locs = torch.full((N, F, C, 2), (P - 1) / 2)
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
    labels = np.zeros((N // 2, F, Q), dtype=[("aoi", int), ("frame", int), ("z", int)])
    labels["aoi"] = np.arange(N // 2).reshape(-1, 1, 1)
    labels["frame"] = np.arange(F).reshape(-1, 1)
    if "alpha" in params:
        # crosstalk simulations
        data[:, :, :] = samples["data"][0].data.floor()
        for q in range(Q):
            labels["z"][:, :, q] = samples[f"z_q{q}"][0][: N // 2].cpu()
    elif "pi" in params:
        data[:, :, :] = samples["data"][0].data.floor()
        labels["z"][:, :, :] = samples["z"][0][: N // 2].cpu()
    else:
        # kinetic simulations
        for f in range(F):
            data[:, f : f + 1, :] = samples[f"data_f{f}"][0].data.floor()
            labels["z"][:, f : f + 1, :] = samples[f"z_f{f}"][0][: N // 2].cpu()

    return CosmosDataset(
        data.cpu(),
        target_locs.cpu(),
        is_ontarget.cpu(),
        labels=labels,
        offset_samples=offset,
        offset_weights=torch.ones(3) / 3,
        device=model.device,
    )
