# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import namedtuple
from pathlib import Path

import torch
from pyro.ops.stats import quantile
from torch.distributions.utils import lazy_property, probs_to_logits


class CosmosData(namedtuple("CosmosData", ["data", "xy", "labels", "device"])):
    @property
    def N(self):
        return self.data.shape[0]

    @property
    def F(self):
        return self.data.shape[1]

    @property
    def P(self):
        return self.data.shape[2]

    @property
    def x(self):
        return self.xy[..., 0]

    @property
    def y(self):
        return self.xy[..., 1]

    @lazy_property
    def median(self):
        return torch.median(self.data).item()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            assert len(idx) == 2
            ndx, fdx = idx
            assert isinstance(fdx, int)
            return self.data[ndx, fdx].to(self.device), self.xy[ndx, fdx].to(
                self.device
            )
        return self.data[idx].to(self.device), self.xy[idx].to(self.device)


class OffsetData(namedtuple("OffsetData", ["data", "device"])):
    @lazy_property
    def median(self):
        return torch.median(self.data).item()

    @lazy_property
    def max(self):
        return quantile(self.data.flatten().float(), 0.995).item()

    @lazy_property
    def min(self):
        return quantile(self.data.flatten().float(), 0.005).item()

    @lazy_property
    def samples(self):
        clamped_offset = torch.clamp(self.data, self.min, self.max)
        offset_samples, offset_weights = torch.unique(
            clamped_offset, sorted=True, return_counts=True
        )
        return offset_samples.to(self.device)

    @lazy_property
    def weights(self):
        clamped_offset = torch.clamp(self.data, self.min, self.max)
        offset_samples, offset_weights = torch.unique(
            clamped_offset, sorted=True, return_counts=True
        )
        return (offset_weights.float() / offset_weights.sum()).to(self.device)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.weights)

    @lazy_property
    def mean(self):
        return torch.sum(self.samples * self.weights).item()

    @lazy_property
    def var(self):
        return torch.sum(self.samples ** 2 * self.weights).item() - self.mean ** 2


class CosmosDataset:
    """
    Cosmos Dataset
    """

    def __init__(
        self,
        ontarget_data,
        ontarget_xy,
        ontarget_labels=None,
        offtarget_data=None,
        offtarget_xy=None,
        offtarget_labels=None,
        offset=None,
        device=torch.device("cpu"),
    ):
        self.ontarget = CosmosData(ontarget_data, ontarget_xy, ontarget_labels, device)
        self.offtarget = CosmosData(
            offtarget_data, offtarget_xy, offtarget_labels, device
        )
        self.offset = OffsetData(offset, device)

    @property
    def P(self):
        return self.ontarget.P

    @lazy_property
    def vmin(self):
        return quantile(self.ontarget.data.flatten().float(), 0.05).item()

    @lazy_property
    def vmax(self):
        return quantile(self.ontarget.data.flatten().float(), 0.99).item()

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self.ontarget.N}, F={self.ontarget.F}, P={self.P})"

    def __str__(self):
        return f"{self.__class__.__name__}(N={self.ontarget.N}, F={self.ontarget.F}, P={self.P})"


def save(obj, path):
    path = Path(path)
    torch.save(
        {
            "ontarget_data": obj.ontarget.data,
            "ontarget_xy": obj.ontarget.xy,
            "ontarget_labels": obj.ontarget.labels,
            "offtarget_data": obj.offtarget.data,
            "offtarget_xy": obj.offtarget.xy,
            "offtarget_labels": obj.offtarget.labels,
            "offset": obj.offset.data,
        },
        path / "data.tpqr",
    )


def load(path, device=torch.device("cpu")):
    path = Path(path)
    data_tapqir = torch.load(path / "data.tpqr")
    return CosmosDataset(**data_tapqir, **{"device": device})
