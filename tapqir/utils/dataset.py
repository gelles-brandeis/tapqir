# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from pathlib import Path

import torch
from pyro.ops.indexing import Vindex
from pyro.ops.stats import quantile
from torch.distributions.utils import lazy_property, probs_to_logits


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
        images,
        xy,
        is_ontarget,
        labels=None,
        offset_samples=None,
        offset_weights=None,
        device=torch.device("cpu"),
        name=None,
    ):
        self.images = images
        self.xy = xy
        self.is_ontarget = is_ontarget
        self.labels = labels
        self.device = device
        self.offset = OffsetData(offset_samples.to(device), offset_weights.to(device))
        self.name = name

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
        return (
            Vindex(self.images)[ndx, fdx, cdx].to(self.device),
            Vindex(self.xy)[ndx, fdx, cdx].to(self.device),
            Vindex(self.is_ontarget)[ndx].to(self.device),
        )

    @lazy_property
    def vmin(self):
        return quantile(self.images.flatten().float(), 0.05).item()

    @lazy_property
    def vmax(self):
        return quantile(self.images.flatten().float(), 0.99).item()

    def __repr__(self):
        samples = repr(self.offset.samples).replace("\n", "\n                  ")
        weights = repr(self.offset.weights).replace("\n", "\n                  ")
        return (
            f"{self.__class__.__name__}: {self.name}"
            f"\n  images           tensor(N={self.N} AOIs, "
            f"F={self.F} frames, "
            f"C={self.C} channels, "
            f"P={self.P} pixels, "
            f"P={self.P} pixels)"
            f"\n  x                tensor(N={self.N} AOIs, "
            f"F={self.F} frames, "
            f"C={self.C} channels)"
            f"\n  y                tensor(N={self.N} AOIs, "
            f"F={self.F} frames, "
            f"C={self.C} channels)"
            f"\n\n  offset.samples   {samples}"
            f"\n        .weights   {weights}"
        )


def save(obj, path):
    path = Path(path)
    torch.save(
        {
            "images": obj.images,
            "xy": obj.xy,
            "is_ontarget": obj.is_ontarget,
            "labels": obj.labels,
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
