from pathlib import Path

import numpy as np
import torch
from pyro.ops.stats import quantile
from torch.distributions.utils import lazy_property, probs_to_logits
from torch.utils.data import Dataset


class CosmosDataset(Dataset):
    """
    Cosmos Dataset
    """

    def __init__(
        self,
        data=None,
        target_locs=None,
        dtype=None,
        device=None,
        labels=None,
        offset=None,
    ):
        self.data = data.to(device)
        self.N, self.F, self.D, _ = self.data.shape
        self.target_locs = target_locs
        self.labels = labels
        if dtype == "test":
            if offset is not None:
                self.offset = offset.to(device)
            else:
                self.offset = offset
        self.dtype = dtype

    @lazy_property
    def vmin(self):
        return np.percentile(self.data.cpu().numpy(), 5)

    @lazy_property
    def vmax(self):
        return np.percentile(self.data.cpu().numpy(), 99)

    @lazy_property
    def data_median(self):
        return torch.median(self.data)

    @lazy_property
    def offset_median(self):
        return torch.median(self.offset)

    @lazy_property
    def noise(self):
        return (
            (self.data.std(dim=(1, 2, 3)).mean() - self.offset.std())
            * np.pi
            * (2 * 1.3) ** 2
        )

    @lazy_property
    def offset_max(self):
        return quantile(self.offset.flatten(), 0.995).item()

    @lazy_property
    def offset_min(self):
        return quantile(self.offset.flatten(), 0.005).item()

    @lazy_property
    def offset_samples(self):
        clamped_offset = torch.clamp(self.offset, self.offset_min, self.offset_max)
        offset_samples, offset_weights = torch.unique(
            clamped_offset, sorted=True, return_counts=True
        )
        return offset_samples

    @lazy_property
    def offset_weights(self):
        clamped_offset = torch.clamp(self.offset, self.offset_min, self.offset_max)
        offset_samples, offset_weights = torch.unique(
            clamped_offset, sorted=True, return_counts=True
        )
        return offset_weights.float() / offset_weights.sum()

    @lazy_property
    def offset_logits(self):
        return probs_to_logits(self.offset_weights)

    @lazy_property
    def offset_mean(self):
        return torch.sum(self.offset_samples * self.offset_weights)

    @lazy_property
    def offset_var(self):
        return (
            torch.sum(self.offset_samples ** 2 * self.offset_weights)
            - self.offset_mean ** 2
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return rf"{self.__class__.__name__}(N={self.N}, F={self.F}, D={self.D}, dtype={self.dtype})"

    def __str__(self):
        return f"{self.__class__.__name__}(N={self.N}, F={self.F}, D={self.D}, dtype={self.dtype})"

    def save(self, path):
        path = Path(path)
        if not path.is_dir():
            path.mkdir()
        torch.save(self.data, path / f"{self.dtype}_data.pt")
        torch.save(self.target_locs, path / f"{self.dtype}_target_locs.pt")
        if self.labels is not None:
            np.save(path / f"{self.dtype}_labels.npy", self.labels)
        if self.dtype == "test":
            if self.offset is not None:
                torch.save(self.offset, path / "offset.pt")


def load_data(path, dtype, device=None):
    path = Path(path)
    data = torch.load(path / f"{dtype}_data.pt", map_location=device).detach()
    target_locs = torch.load(
        path / f"{dtype}_target_locs.pt", map_location=device
    ).detach()
    labels = None
    if (path / f"{dtype}_labels.npy").is_file():
        labels = np.load(path / f"{dtype}_labels.npy")
    if dtype == "test":
        offset = torch.load(path / "offset.pt", map_location=device).detach()
        return CosmosDataset(data, target_locs, dtype, device, labels, offset)

    return CosmosDataset(data, target_locs, dtype, device, labels)
