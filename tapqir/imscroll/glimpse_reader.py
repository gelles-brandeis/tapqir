# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tqdm import tqdm

from tapqir.utils.dataset import CosmosDataset, save


class GlimpseDataset:
    """
    GlimpseDataset parses header, aoiinfo, driftlist, and intervals (optional)
    files and creates 1) aoiinfo and cumdrift DataFrames, 2) __getitem__ method
    to retrieve glimpse image for a given frame, 3) labels np.array.

    :param path: path to the folder containing options.cfg file.
    """

    def __init__(self, path):
        """Read Glimpse files"""

        path = Path(path)
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = path / "options.cfg"
        config.read(cfg_file)

        dtypes = ["ontarget"]
        if config["glimpse"]["offtarget_aoiinfo"] is not None:
            dtypes.append("offtarget")

        # convert header into dict format
        mat_header = loadmat(Path(config["glimpse"]["dir"]) / "header.mat")
        header = dict()
        for i, dt in enumerate(mat_header["vid"].dtype.names):
            header[dt] = np.squeeze(mat_header["vid"][0, 0][i])

        # load driftlist mat file
        drift_mat = loadmat(config["glimpse"]["driftlist"])
        # convert driftlist into DataFrame
        drift_df = pd.DataFrame(
            drift_mat["driftlist"][:, :3], columns=["frame", "dx", "dy"]
        )
        drift_df = drift_df.astype({"frame": int}).set_index("frame")

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
            # adjust to python indexing
            aoi_df[dtype]["x"] = aoi_df[dtype]["x"] - 1
            aoi_df[dtype]["y"] = aoi_df[dtype]["y"] - 1

        # calculate the cumulative sum of dx and dy relative to the aoiinfo frame
        aoiinfo_frame = int(aoi_df["ontarget"].at[1, "frame"])
        drift_df.loc[aoiinfo_frame + 1 :] = drift_df.loc[aoiinfo_frame + 1 :].cumsum(
            axis=0
        )
        drift_df.loc[aoiinfo_frame - 1 :: -1] = (
            -drift_df.loc[aoiinfo_frame : drift_df.index[1] : -1]
        ).cumsum(axis=0)

        if config["glimpse"]["frame_start"] and config["glimpse"]["frame_end"]:
            f1 = int(config["glimpse"]["frame_start"])
            f2 = int(config["glimpse"]["frame_end"])
            drift_df = drift_df.loc[f1:f2]

        labels = defaultdict(lambda: None)
        for dtype in dtypes:
            if config["glimpse"][f"{dtype}_labels"] is not None:
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
                spot_picker = labels_mat["Intervals"]["CumulativeIntervalArray"][0, 0]
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
        self.aoiinfo = aoi_df
        self.cumdrift = drift_df
        self.labels = labels

    def __len__(self):
        return self.N

    def __getitem__(self, key):
        # read the entire frame image
        if isinstance(key, slice):
            imgs = []
            for frame in range(
                key.start, key.stop, 1 if key.step is None else key.step
            ):
                imgs.append(self[frame])
            return np.stack(imgs, 0)
        frame = key
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


def read_glimpse(path, P):
    glimpse = GlimpseDataset(path)

    abs_locs = defaultdict(lambda: None)
    target_xy = defaultdict(lambda: None)
    data = defaultdict(lambda: None)
    for dtype in glimpse.dtypes:
        abs_locs[dtype] = (
            np.expand_dims(glimpse.aoiinfo[dtype][["x", "y"]].values, axis=1)
            + glimpse.cumdrift[["dx", "dy"]].values
        )
        target_xy[dtype] = np.zeros(
            (len(glimpse.aoiinfo[dtype]), len(glimpse.cumdrift), 2)
        )
        data[dtype] = np.zeros(
            (len(glimpse.aoiinfo[dtype]), len(glimpse.cumdrift), P, P), dtype="int"
        )
    offsets = np.zeros((len(glimpse.cumdrift), 4, 30, 30), dtype="int")
    # loop through each frame
    for i, frame in enumerate(tqdm(glimpse.cumdrift.index)):
        img = glimpse[frame]

        offsets[i, 0, :, :] += img[10:40, 10:40]
        offsets[i, 1, :, :] += img[10:40, -40:-10]
        offsets[i, 2, :, :] += img[-40:-10, 10:40]
        offsets[i, 3, :, :] += img[-40:-10, -40:-10]
        for dtype in glimpse.dtypes:
            # loop through each aoi
            for j, aoi in enumerate(glimpse.aoiinfo[dtype].index):
                top_x = round(abs_locs[dtype][j, i, 0] - 0.5 * (P - 1))
                left_y = round(abs_locs[dtype][j, i, 1] - 0.5 * (P - 1))
                data[dtype][j, i, :, :] += img[top_x : top_x + P, left_y : left_y + P]
                target_xy[dtype][j, i, 0] = abs_locs[dtype][j, i, 0] - top_x
                target_xy[dtype][j, i, 1] = abs_locs[dtype][j, i, 1] - left_y
    offset = torch.tensor(offsets)
    for dtype in glimpse.dtypes:
        assert (target_xy[dtype] > 0.5 * P - 1).all()
        assert (target_xy[dtype] < 0.5 * P).all()
        # convert data into torch tensor
        data[dtype] = torch.tensor(data[dtype])
        target_xy[dtype] = torch.tensor(target_xy[dtype])
    dataset = CosmosDataset(
        data["ontarget"],
        target_xy["ontarget"],
        glimpse.labels["ontarget"],
        data["offtarget"],
        target_xy["offtarget"],
        glimpse.labels["offtarget"],
        offset,
    )
    save(dataset, path)
