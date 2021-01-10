import numpy as np
import pandas as pd
import pyro
import torch
from pyro import param, poutine
from pyro.infer import Predictive
from torch.distributions import constraints

from tapqir.models import Cosmos, GaussianSpot
from tapqir.utils.dataset import CosmosDataset


def simulate(N, F, D=14, cuda=True, params=dict()):
    pyro.set_rng_seed(0)
    pyro.get_param_store().clear()
    S, K = 1, 2

    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        device = torch.device("cpu")

    # parameters and samples
    param("gain", torch.tensor(params["gain"]), constraint=constraints.positive)
    param("width_mean", torch.tensor(1.5), constraint=constraints.positive)
    param("width_size", torch.tensor(2.0), constraint=constraints.positive)
    param(
        "probs_z",
        torch.tensor([1 - params["probs_z"], params["probs_z"]]),
        constraint=constraints.simplex,
    )
    param("rate_j", torch.tensor(params["rate_j"]), constraint=constraints.positive)
    param(
        "proximity",
        torch.tensor([params["proximity"]]),
        constraint=constraints.positive,
    )

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

    model = Cosmos(S, K)
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
        poutine.uncondition(model.model), posterior_samples=samples, num_samples=None
    )
    samples = predictive()
    model.data.data = samples["d/data"][0].data.floor()
    model.control.data = samples["c/data"][0].data.floor()
    model.data.labels = np.zeros(
        (N, F), dtype=[("aoi", int), ("frame", int), ("z", bool)]
    )
    model.data.labels["aoi"] = np.arange(N).reshape(-1, 1)
    model.data.labels["frame"] = np.arange(F)
    model.data.labels["z"] = samples["d/theta"][0].cpu() > 0

    return model
