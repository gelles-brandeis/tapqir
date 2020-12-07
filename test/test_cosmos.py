import torch
import pyro
from pyro import param, poutine
from pyro.infer import Predictive
from torch.distributions import constraints
import numpy as np
import pandas as pd
import os

from tapqir.utils.dataset import CosmosDataset
from tapqir.models import GaussianSpot, Cosmos


def test_cosmos():
    smoke_test = ("CI" in os.environ)
    if smoke_test:
        torch.set_default_tensor_type("torch.FloatTensor")
        device = torch.device("cpu")
        device_str = "cpu"
    else:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda")
        device_str = "cuda"
    S, K = 1, 2
    N = 5  # number of AOIs
    D = 14  # AOI size
    F = 500  # number of frames
    offset = torch.full((1,), 90.)
    target = pd.DataFrame(data={"frame": np.zeros(N), "x": 6.5, "y": 6.5}, index=np.arange(N))
    target.index.name = "aoi"
    drift = pd.DataFrame(data={"dx": 0., "dy": 0}, index=np.arange(F))
    drift.index.name = "frame"
    data = CosmosDataset(torch.zeros(N, F, D, D), target, drift,
                         dtype="test", device=device, offset=offset)
    control = CosmosDataset(torch.zeros(N, F, D, D), target, drift,
                            dtype="control", device=device)

    pyro.set_rng_seed(0)
    pyro.get_param_store().clear()
    param("gain", torch.tensor(7.), constraint=constraints.positive)
    param("height_loc", torch.tensor(3000.), constraint=constraints.positive)
    param("width_mean", torch.tensor(1.5), constraint=constraints.positive)
    param("width_size", torch.tensor(2.), constraint=constraints.positive)
    param("probs_z", torch.tensor([0.85, 0.15]), constraint=constraints.simplex)
    param("rate_j", torch.tensor(0.5), constraint=constraints.positive)
    param("proximity", torch.tensor([0.2]), constraint=constraints.positive)
    model = Cosmos(S, K)
    model.data = data
    model.data_loc = GaussianSpot(model.data.target, model.data.drift, model.data.D)
    model.control = control
    model.control_loc = GaussianSpot(model.control.target, model.control.drift, model.control.D)
    model.size = torch.cat((torch.tensor([2.]), (((D+1)/(2*param("proximity")))**2-1)), dim=-1)
    simulation = {
        "d/background": torch.full((1, N, 1), 150.),
        "d/width": torch.full((1, 2, N, F), 1.4),
        "c/background": torch.full((1, N, 1), 150.),
        "c/width": torch.full((1, 2, N, F), 1.4),
    }
    predictive = Predictive(poutine.uncondition(model.model), posterior_samples=simulation, num_samples=None)
    samples = predictive()
    model.data.data = samples["d/data"][0].data.floor()
    model.control.data = samples["c/data"][0].data.floor()
    model.data.labels = np.zeros((N, F), dtype=[("aoi", int), ("frame", int), ("z", bool)])
    model.data.labels["aoi"] = np.arange(N).reshape(-1, 1)
    model.data.labels["frame"] = np.arange(F)
    model.data.labels["z"] = samples["d/theta"][0].cpu() > 0
    path_data = "simulation"
    model.data.save(path_data)
    model.control.save(path_data)

    learning_rate = 0.005
    batch_size = 2
    num_iter = 200 if smoke_test else 200
    infer = 200 if smoke_test else 200
    model.load(path_data, False, device_str)
    model.settings(learning_rate, batch_size)
    model.run(num_iter, infer)
