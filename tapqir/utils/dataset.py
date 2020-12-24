import configparser
import os

import numpy as np
import pandas as pd
import torch
from pyro.ops.stats import quantile
from scipy.io import loadmat
from torch.distributions.utils import lazy_property, probs_to_logits
from torch.utils.data import Dataset


class CosmosDataset(Dataset):
    """
    Cosmos Dataset
    """

    def __init__(
        self,
        data=None,
        target=None,
        drift=None,
        dtype=None,
        device=None,
        offset=None,
        labels=None,
    ):
        self.data = data.to(device)
        self.N, self.F, self.D, _ = self.data.shape
        self.target = target
        self.drift = drift
        if dtype == "test":
            self.labels = labels
            if offset is not None:
                self.offset = offset.to(device)
            else:
                self.offset = offset
        assert self.N == len(self.target)
        assert self.F == len(self.drift)
        self.dtype = dtype
        self.vmin = np.percentile(self.data.cpu().numpy(), 5)
        self.vmax = np.percentile(self.data.cpu().numpy(), 99)

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
        return (
            f"{self.__class__.__name__}"
            + f"\nN={self.N!r}, F={self.F!r}, D={self.D!r}"
            + f"\ndtype={self.dtype!r}"
        )

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(self.data, os.path.join(path, "{}_data.pt".format(self.dtype)))
        self.target.to_csv(os.path.join(path, "{}_target.csv".format(self.dtype)))
        self.drift.to_csv(os.path.join(path, "drift.csv"))
        if self.dtype == "test":
            if self.offset is not None:
                torch.save(self.offset, os.path.join(path, "offset.pt"))
            if self.labels is not None:
                np.save(os.path.join(path, "labels.npy"), self.labels)


def load_data(path, dtype, device=None):
    data = torch.load(
        os.path.join(path, "{}_data.pt".format(dtype)), map_location=device
    ).detach()
    target = pd.read_csv(
        os.path.join(path, "{}_target.csv".format(dtype)), index_col="aoi"
    )
    drift = pd.read_csv(os.path.join(path, "drift.csv"), index_col="frame")
    if dtype == "test":
        offset = torch.load(
            os.path.join(path, "offset.pt"), map_location=device
        ).detach()
        labels = None
        if os.path.isfile(os.path.join(path, "labels.npy")):
            labels = np.load(os.path.join(path, "labels.npy"))
        return CosmosDataset(data, target, drift, dtype, device, offset, labels)

    return CosmosDataset(data, target, drift, dtype, device)


class GlimpseDataset(Dataset):
    """
    Glimpse Dataset
    """

    def __init__(self, path):
        # read options.cfg file
        self.config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = os.path.join(path, "options.cfg")
        self.config.read(cfg_file)
        # convert header into dict format
        mat_header = loadmat(os.path.join(self.config["glimpse"]["dir"], "header.mat"))
        self.header = dict()
        for i, dt in enumerate(mat_header["vid"].dtype.names):
            self.header[dt] = np.squeeze(mat_header["vid"][0, 0][i])
        self.height, self.width = int(self.header["height"]), int(self.header["width"])

    def __len__(self):
        return self.N

    def __getitem__(self, frame):
        # read the entire frame image
        glimpse_number = self.header["filenumber"][frame - 1]
        glimpse_path = os.path.join(
            self.config["glimpse"]["dir"], "{}.glimpse".format(glimpse_number)
        )
        offset = self.header["offset"][frame - 1]
        with open(glimpse_path, "rb") as fid:
            fid.seek(offset)
            img = np.fromfile(fid, dtype=">i2", count=self.height * self.width).reshape(
                self.height, self.width
            )
        return img + 2 ** 15

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            + f"\nN={self.N!r}, F={self.F!r}, D={self.D!r}"
            + f"\ndtype={self.dtype!r}"
        )

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(self.data, os.path.join(path, "{}_data.pt".format(self.dtype)))
        self.target.to_csv(os.path.join(path, "{}_target.csv".format(self.dtype)))
        self.drift.to_csv(os.path.join(path, "drift.csv"))
        if self.dtype == "test":
            if self.offset is not None:
                torch.save(self.offset, os.path.join(path, "offset.pt"))
            if self.labels is not None:
                np.save(os.path.join(path, "labels.npy"), self.labels)
