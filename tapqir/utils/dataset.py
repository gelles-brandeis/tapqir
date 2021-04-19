import configparser
from collections import defaultdict
from pathlib import Path

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
        labels=None,
        offset=None,
    ):
        self.data = data.to(device)
        self.N, self.F, self.D, _ = self.data.shape
        self.target = target
        self.drift = drift
        self.labels = labels
        if dtype == "test":
            if offset is not None:
                self.offset = offset.to(device)
            else:
                self.offset = offset
        assert self.N == len(self.target)
        assert self.F == len(self.drift)
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
        self.target.to_csv(path / f"{self.dtype}_target.csv")
        self.drift.to_csv(path / "drift.csv")
        if self.labels is not None:
            np.save(path / f"{self.dtype}_labels.npy", self.labels)
        if self.dtype == "test":
            if self.offset is not None:
                torch.save(self.offset, path / "offset.pt")


def load_data(path, dtype, device=None):
    path = Path(path)
    data = torch.load(path / f"{dtype}_data.pt", map_location=device).detach()
    target = pd.read_csv(path / f"{dtype}_target.csv", index_col="aoi")
    drift = pd.read_csv(path / "drift.csv", index_col="frame")
    labels = None
    if (path / f"{dtype}_labels.npy").is_file():
        labels = np.load(path / f"{dtype}_labels.npy")
    if dtype == "test":
        offset = torch.load(path / "offset.pt", map_location=device).detach()
        return CosmosDataset(data, target, drift, dtype, device, labels, offset)

    return CosmosDataset(data, target, drift, dtype, device, labels)


class GlimpseDataset(Dataset):
    """
    Glimpse Dataset
    """

    def __init__(self, path):
        """ Read Glimpse files """

        path = Path(path)
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = path / "options.cfg"
        config.read(cfg_file)

        dtypes = ["test"]
        if config["glimpse"]["control_aoiinfo"] is not None:
            dtypes.append("control")

        # convert header into dict format
        mat_header = loadmat(Path(config["glimpse"]["dir"]) / "header.mat")
        header = dict()
        for i, dt in enumerate(mat_header["vid"].dtype.names):
            header[dt] = np.squeeze(mat_header["vid"][0, 0][i])

        # load driftlist mat file
        drift_mat = loadmat(config["glimpse"]["driftlist"])
        # calculate the cumulative sum of dx and dy
        drift_mat["driftlist"][:, 1:3] = np.cumsum(
            drift_mat["driftlist"][:, 1:3], axis=0
        )
        # convert driftlist into DataFrame
        drift_df = pd.DataFrame(
            drift_mat["driftlist"][:, :3], columns=["frame", "dx", "dy"]
        )
        drift_df = drift_df.astype({"frame": int}).set_index("frame")

        if config["glimpse"]["frame_start"] and config["glimpse"]["frame_end"]:
            f1 = int(config["glimpse"]["frame_start"])
            f2 = int(config["glimpse"]["frame_end"])
            drift_df = drift_df.loc[f1:f2]

        # load aoiinfo mat file
        aoi_mat = {}
        aoi_df = {}
        for dtype in dtypes:
            try:
                aoi_mat[dtype] = loadmat(config["glimpse"][f"{dtype}_aoiinfo"])
            except ValueError:
                aoi_mat[dtype] = np.loadtxt(config["glimpse"][f"{dtype}_aoiinfo"])
            try:
                aoi_df[dtype] = pd.DataFrame(
                    aoi_mat[dtype]["aoiinfo2"],
                    columns=["frame", "ave", "x", "y", "pixnum", "aoi"],
                )
            except KeyError:
                aoi_df[dtype] = pd.DataFrame(
                    aoi_mat[dtype]["aoifits"]["aoiinfo2"][0, 0],
                    columns=["frame", "ave", "x", "y", "pixnum", "aoi"],
                )
            except IndexError:
                aoi_df[dtype] = pd.DataFrame(
                    aoi_mat[dtype], columns=["frame", "ave", "x", "y", "pixnum", "aoi"]
                )
            aoi_df[dtype] = aoi_df[dtype].astype({"aoi": int}).set_index("aoi")
            aoi_df[dtype]["x"] = (
                aoi_df[dtype]["x"]
                - drift_df.at[int(aoi_df[dtype].at[1, "frame"]), "dx"]
            )
            aoi_df[dtype]["y"] = (
                aoi_df[dtype]["y"]
                - drift_df.at[int(aoi_df[dtype].at[1, "frame"]), "dy"]
            )

        labels = defaultdict(None)
        for dtype in dtypes:
            if config["glimpse"][f"{dtype}_labels"] is not None:
                if config["glimpse"]["labeltype"] is not None:
                    framelist = loadmat(config["glimpse"][f"{dtype}_labels"])
                    f1 = framelist[config["glimpse"]["labeltype"]][0, 2]
                    f2 = framelist[config["glimpse"]["labeltype"]][-1, 2]
                    drift_df = drift_df.loc[f1:f2]
                    aoi_list = np.unique(
                        framelist[config["glimpse"]["labeltype"]][:, 0]
                    )
                    aoi_df["test"] = aoi_df["test"].loc[aoi_list]
                    labels[dtype] = np.zeros(
                        (len(aoi_df[dtype]), len(drift_df)),
                        dtype=[
                            ("aoi", int),
                            ("frame", int),
                            ("z", int),
                            ("spotpicker", int),
                        ],
                    )
                    labels[dtype]["aoi"] = framelist[config["glimpse"]["labeltype"]][
                        :, 0
                    ].reshape(len(aoi_df[dtype]), len(drift_df))
                    labels[dtype]["frame"] = framelist[config["glimpse"]["labeltype"]][
                        :, 2
                    ].reshape(len(aoi_df[dtype]), len(drift_df))
                    labels["spotpicker"] = framelist[config["glimpse"]["labeltype"]][
                        :, 1
                    ].reshape(len(aoi_df[dtype]), len(drift_df))
                    labels[dtype]["spotpicker"][labels[dtype]["spotpicker"] == 0] = 3
                    labels[dtype]["spotpicker"][labels[dtype]["spotpicker"] == 2] = 0
                else:
                    labels_mat = loadmat(config["glimpse"][f"{dtype}_labels"])
                    labels[dtype] = np.zeros(
                        (len(aoi_df[dtype]), len(drift_df)),
                        dtype=[
                            ("aoi", int),
                            ("frame", int),
                            ("z", bool),
                            ("spotpicker", float),
                        ],
                    )
                    labels[dtype]["aoi"] = aoi_df[dtype].index.values.reshape(-1, 1)
                    labels[dtype]["frame"] = drift_df.index.values
                    spot_picker = labels_mat["Intervals"]["CumulativeIntervalArray"][
                        0, 0
                    ]
                    for sp in spot_picker:
                        aoi = int(sp[-1])
                        start = int(sp[1])
                        end = int(sp[2])
                        if sp[0] in [-2.0, 0.0, 2.0]:
                            labels[dtype]["spotpicker"][
                                (labels[dtype]["aoi"] == aoi)
                                & (labels[dtype]["frame"] >= start)
                                & (labels[dtype]["frame"] <= end)
                            ] = 0
                        elif sp[0] in [-3.0, 1.0, 3.0]:
                            labels[dtype]["spotpicker"][
                                (labels[dtype]["aoi"] == aoi)
                                & (labels[dtype]["frame"] >= start)
                                & (labels[dtype]["frame"] <= end)
                            ] = 1

                labels[dtype]["z"] = labels[dtype]["spotpicker"]

        self.height, self.width = int(header["height"]), int(header["width"])
        self.config = config
        self.header = header
        self.dtypes = dtypes
        self.aoi_df = aoi_df
        self.drift_df = drift_df
        self.labels = labels

    def __len__(self):
        return self.N

    def __getitem__(self, frame):
        # read the entire frame image
        glimpse_number = self.header["filenumber"][frame - 1]
        glimpse_path = Path(self.config["glimpse"]["dir"]) / f"{glimpse_number}.glimpse"
        offset = self.header["offset"][frame - 1]
        with open(glimpse_path, "rb") as fid:
            fid.seek(offset)
            img = np.fromfile(fid, dtype=">i2", count=self.height * self.width).reshape(
                self.height, self.width
            )
        return img + 2 ** 15

    def __repr__(self):
        return rf"{self.__class__.__name__}(N={self.N}, F={self.F}, D={self.D}, dtype={self.dtype})"

    def __str__(self):
        return f"{self.__class__.__name__}(N={self.N}, F={self.F}, D={self.D}, dtype={self.dtype})"

    def save(self, path):
        path = Path(path)
        if not path.is_dir():
            path.mkdir()
        torch.save(self.data, path / f"{self.dtype}_data.pt")
        self.target.to_csv(path / f"{self.dtype}_target.csv")
        self.drift.to_csv(path / "drift.csv")
        if self.dtype == "test":
            if self.offset is not None:
                torch.save(self.offset, path / "offset.pt")
            if self.labels is not None:
                np.save(path / "labels.npy", self.labels)
