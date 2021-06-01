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

    def fetch(self, ndx, fdx=None):
        if fdx is not None:
            assert isinstance(fdx, int)
            return self.data[ndx, fdx].to(self.device), self.xy[ndx, fdx].to(
                self.device
            )
        return self.data[ndx].to(self.device), self.xy[ndx].to(self.device)


class OffsetData(namedtuple("OffsetData", ["samples", "weights"])):
    @lazy_property
    def min(self):
        return torch.min(self.samples).item()

    @lazy_property
    def max(self):
        return torch.max(self.samples).item()

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
        offset_samples=None,
        offset_weights=None,
        device=torch.device("cpu"),
        title=None,
    ):
        self.ontarget = CosmosData(ontarget_data, ontarget_xy, ontarget_labels, device)
        self.offtarget = CosmosData(
            offtarget_data, offtarget_xy, offtarget_labels, device
        )
        self.offset = OffsetData(offset_samples.to(device), offset_weights.to(device))
        self.title = title

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
        samples = repr(self.offset.samples).replace("\n", "\n                  ")
        weights = repr(self.offset.weights).replace("\n", "\n                  ")
        return (
            f"{self.__class__.__name__}: {self.title}"
            f"\n  ontarget.data   tensor(N={self.ontarget.N} AOIs, "
            f"F={self.ontarget.F} frames, "
            f"P={self.ontarget.P} pixels, "
            f"P={self.ontarget.P} pixels)"
            f"\n          .x      tensor(N={self.ontarget.N} AOIs, "
            f"F={self.ontarget.F} frames)"
            f"\n          .y      tensor(N={self.ontarget.N} AOIs, "
            f"F={self.ontarget.F} frames)"
            f"\n\n  offtarget.data  tensor(N={self.offtarget.N} AOIs, "
            f"F={self.offtarget.F} frames, "
            f"P={self.offtarget.P} pixels, "
            f"P={self.offtarget.P} pixels)"
            f"\n           .x     tensor(N={self.offtarget.N} AOIs, "
            f"F={self.offtarget.F} frames)"
            f"\n           .y     tensor(N={self.offtarget.N} AOIs, "
            f"F={self.offtarget.F} frames)"
            f"\n\n  offset.samples  {samples}"
            f"\n        .weights  {weights}"
        )


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
            "offset_samples": obj.offset.samples,
            "offset_weights": obj.offset.weights,
            "title": obj.title,
        },
        path / "data.tpqr",
    )


def load(path, device=torch.device("cpu")):
    path = Path(path)
    data_tapqir = torch.load(path / "data.tpqr")
    return CosmosDataset(**data_tapqir, **{"device": device})
