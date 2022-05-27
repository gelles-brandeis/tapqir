# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import namedtuple
from pathlib import Path

import torch
from pyro.ops.indexing import Vindex
from pyro.ops.stats import quantile
from torch.distributions.utils import lazy_property, probs_to_logits

from tapqir.exceptions import TapqirFileNotFoundError

logger = logging.getLogger(__name__)


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
        return torch.sum(self.samples**2 * self.weights).item() - self.mean**2


class CosmosDataset:
    """
    Cosmos Dataset
    """

    def __init__(
        self,
        images,
        xy,
        is_ontarget,
        mask=None,
        labels=None,
        offset_samples=None,
        offset_weights=None,
        device=torch.device("cpu"),
        time1=None,
        ttb=None,
        name=None,
    ):
        self.images = images
        self.xy = xy
        self.is_ontarget = is_ontarget
        if mask is None:
            mask = torch.ones_like(is_ontarget, dtype=torch.bool)
        self.mask = mask
        self.labels = labels
        self.device = device
        self.offset = OffsetData(offset_samples.to(device), offset_weights.to(device))
        self.time1 = time1
        self.ttb = ttb
        self.name = name

    @lazy_property
    def N(self) -> int:
        """
        Number of on-target AOIs.
        """
        return self.is_ontarget.sum().item()

    @lazy_property
    def Nc(self) -> int:
        """
        Number of off-target AOIs.
        """
        return (~self.is_ontarget).sum().item()

    @lazy_property
    def Nt(self) -> int:
        """
        Total number of AOIs.
        """
        return self.N + self.Nc

    @property
    def F(self) -> int:
        """
        Number of frames.
        """
        return self.images.shape[1]

    @property
    def C(self) -> int:
        """
        Number of color-channels.
        """
        return self.images.shape[2]

    @property
    def P(self) -> int:
        """
        Number of pixels.
        """
        Px, Py = self.images.shape[3], self.images.shape[4]
        assert Px == Py
        return Px

    @property
    def x(self) -> torch.Tensor:
        """
        Target location on the x-axis.
        """
        return self.xy[..., 0]

    @property
    def y(self) -> torch.Tensor:
        """
        Target location on the y-axis.
        """
        return self.xy[..., 1]

    @lazy_property
    def median(self) -> torch.Tensor:
        return torch.stack(
            [torch.median(self.images[..., c, :, :]) for c in range(self.C)]
        )

    def fetch(self, ndx, fdx, cdx):
        return (
            Vindex(self.images)[ndx, fdx, cdx].to(self.device),
            Vindex(self.xy)[ndx, fdx, cdx].to(self.device),
            Vindex(self.is_ontarget)[ndx].to(self.device),
        )

    @lazy_property
    def vmin(self) -> torch.Tensor:
        return torch.stack(
            [
                quantile(self.images[..., c, :, :].flatten().float(), 0.05)
                for c in range(self.C)
            ]
        )

    @lazy_property
    def vmax(self) -> int:
        return torch.stack(
            [
                quantile(self.images[..., c, :, :].flatten().float(), 0.99)
                for c in range(self.C)
            ]
        )

    def __repr__(self):
        samples = repr(self.offset.samples).replace("\n", "\n                  ")
        weights = repr(self.offset.weights).replace("\n", "\n                  ")
        return (
            f"{self.__class__.__name__}: {self.name}"
            f"\n  images           tensor(N={self.N} on-target AOIs, "
            f"Nc={self.Nc} off-target AOIs, "
            f"F={self.F} frames, "
            f"C={self.C} channels, "
            f"P={self.P} pixels, "
            f"P={self.P} pixels)"
            f"\n  x                tensor(N={self.N} on-target AOIs, "
            f"Nc={self.Nc} off-target AOIs, "
            f"F={self.F} frames, "
            f"C={self.C} channels)"
            f"\n  y                tensor(N={self.N} on-target AOIs, "
            f"Nc={self.Nc} off-target AOIs, "
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
            "mask": obj.mask,
            "labels": obj.labels,
            "offset_samples": obj.offset.samples,
            "offset_weights": obj.offset.weights,
            "name": obj.name,
            "time1": obj.time1,
            "ttb": obj.ttb,
        },
        path / "data.tpqr",
    )
    logger.info(f"Data is saved in {path / 'data.tpqr'}")


def load(path, device=torch.device("cpu")):
    path = Path(path)
    try:
        data_tapqir = torch.load(path / "data.tpqr")
    except FileNotFoundError:
        raise TapqirFileNotFoundError("data", path / "data.tpqr")
    return CosmosDataset(**data_tapqir, **{"device": device})
