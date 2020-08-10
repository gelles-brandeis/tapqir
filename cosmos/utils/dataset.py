import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CoSMoSDataset(Dataset):
    """
    CoSMoS Dataset
    """

    def __init__(self, data=None, target=None, drift=None, dtype=None, device=None, offset=None, labels=None):
        self.data = data.to(device)
        self.N, self.F, self.D, _ = self.data.shape
        self.target = target
        self.drift = drift
        if dtype == "test":
            self.labels = labels
            self.offset = offset.to(device)
        assert self.N == len(self.target)
        assert self.F == len(self.drift)
        self.dtype = dtype
        self.vmin = np.percentile(self.data.cpu().numpy(), 5)
        self.vmax = np.percentile(self.data.cpu().numpy(), 99)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               + f"\nN={self.N!r}, F={self.F!r}, D={self.D!r}" \
               + f"\ndtype={self.dtype!r}"

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(self.data, os.path.join(
            path, "{}_data.pt".format(self.dtype)))
        self.target.to_csv(os.path.join(
            path, "{}_target.csv".format(self.dtype)))
        self.drift.to_csv(os.path.join(
            path, "drift.csv"))
        if self.dtype == "test":
            torch.save(self.offset, os.path.join(
                path, "offset.pt"))
            if self.labels is not None:
                np.save(os.path.join(path, "labels.npy"),
                        self.labels)


def load_data(path, dtype, device=None):
    data = torch.load(os.path.join(
        path, "{}_data.pt".format(dtype)),
        map_location=device).detach()
    target = pd.read_csv(os.path.join(
        path, "{}_target.csv".format(dtype)),
        index_col="aoi")
    drift = pd.read_csv(os.path.join(
        path, "drift.csv"),
        index_col="frame")
    if dtype == "test":
        offset = torch.load(os.path.join(
            path, "offset.pt"),
            map_location=device).detach()
        labels = None
        if os.path.isfile(os.path.join(path, "labels.npy")):
            labels = np.load(os.path.join(
                path, "labels.npy"))
        return CoSMoSDataset(data, target, drift, dtype, device, offset, labels)

    return CoSMoSDataset(data, target, drift, dtype, device)
