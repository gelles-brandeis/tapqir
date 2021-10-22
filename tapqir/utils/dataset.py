# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from pathlib import Path

import torch
from pyro.ops.indexing import Vindex
from pyro.ops.stats import quantile
from torch.distributions.utils import lazy_property, probs_to_logits


class CosmosData(namedtuple("CosmosData", ["images", "xy", "labels", "device"])):
    @property
    def N(self):
        return self.images.shape[0]

    @property
    def F(self):
        return self.images.shape[1]

    @property
    def C(self):
        return self.images.shape[2]

    @property
    def P(self):
        return self.images.shape[3]

    @property
    def x(self):
        return self.xy[..., 0]

    @property
    def y(self):
        return self.xy[..., 1]

    @lazy_property
    def median(self):
        return torch.median(self.images).item()

    def fetch(self, ndx, fdx, cdx):
        return Vindex(self.images)[ndx, fdx, cdx].to(self.device), Vindex(self.xy)[
            ndx, fdx, cdx
        ].to(self.device)


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
        name=None,
    ):
        self.ontarget = CosmosData(ontarget_data, ontarget_xy, ontarget_labels, device)
        self.offtarget = CosmosData(
            offtarget_data, offtarget_xy, offtarget_labels, device
        )
        self.offset = OffsetData(offset_samples.to(device), offset_weights.to(device))
        self.name = name

    @property
    def P(self):
        return self.ontarget.P

    @property
    def C(self):
        return self.ontarget.C

    @lazy_property
    def vmin(self):
        return quantile(self.ontarget.images.flatten().float(), 0.05).item()

    @lazy_property
    def vmax(self):
        return quantile(self.ontarget.images.flatten().float(), 0.99).item()

    def __repr__(self):
        samples = repr(self.offset.samples).replace("\n", "\n                  ")
        weights = repr(self.offset.weights).replace("\n", "\n                  ")
        return (
            f"{self.__class__.__name__}: {self.title}"
            f"\n  ontarget.images  tensor(N={self.ontarget.N} AOIs, "
            f"F={self.ontarget.F} frames, "
            f"P={self.ontarget.P} pixels, "
            f"P={self.ontarget.P} pixels)"
            f"\n          .x       tensor(N={self.ontarget.N} AOIs, "
            f"F={self.ontarget.F} frames)"
            f"\n          .y       tensor(N={self.ontarget.N} AOIs, "
            f"F={self.ontarget.F} frames)"
            f"\n\n  offtarget.images tensor(N={self.offtarget.N} AOIs, "
            f"F={self.offtarget.F} frames, "
            f"P={self.offtarget.P} pixels, "
            f"P={self.offtarget.P} pixels)"
            f"\n           .x      tensor(N={self.offtarget.N} AOIs, "
            f"F={self.offtarget.F} frames)"
            f"\n           .y      tensor(N={self.offtarget.N} AOIs, "
            f"F={self.offtarget.F} frames)"
            f"\n\n  offset.samples   {samples}"
            f"\n        .weights   {weights}"
        )


def save(obj, path):
    path = Path(path)
    torch.save(
        {
            "ontarget_data": obj.ontarget.images,
            "ontarget_xy": obj.ontarget.xy,
            "ontarget_labels": obj.ontarget.labels,
            "offtarget_data": obj.offtarget.images,
            "offtarget_xy": obj.offtarget.xy,
            "offtarget_labels": obj.offtarget.labels,
            "offset_samples": obj.offset.samples,
            "offset_weights": obj.offset.weights,
            "name": obj.name,
        },
        path / "data.tpqr",
    )


def load(path, device=torch.device("cpu")):
    path = Path(path)
    data_tapqir = torch.load(path / "data.tpqr")
    return CosmosDataset(**data_tapqir, **{"device": device})
