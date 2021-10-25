# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from scipy.io import loadmat
from tqdm import tqdm

from tapqir.utils.dataset import CosmosDataset, save

# logger
logger = logging.getLogger(__name__)


class GlimpseDataset:
    """
    GlimpseDataset parses header, aoiinfo, driftlist, and intervals (optional)
    files.

    :param name: Channel name.
    :param glimpse-folder: Path to the header/glimpse folder.
    :param ontarget-aoiinfo: Path to the on-target AOI locations file.
    :param offtarget-aoiinfo: Path to the off-target control AOI locations file (optional).
    :param driftlist: Path to the driftlist file.
    :param frame-start: First frame to include in the analysis (optional).
    :param frame-end: Last frame to include in the analysis (optional).
    :param ontarget-labels: Path to the on-target label intervals file.
    :param offtarget-labels: Path to the off-target label intervals file.
    """

    def __init__(self, **kwargs):
        dtypes = ["ontarget"]
        if "offtarget-aoiinfo" in kwargs:
            dtypes.append("offtarget")

        # convert header into dict format
        mat_header = loadmat(Path(kwargs["glimpse-folder"]) / "header.mat")
        header = dict()
        for i, dt in enumerate(mat_header["vid"].dtype.names):
            header[dt] = np.squeeze(mat_header["vid"][0, 0][i])

        # load driftlist mat file
        drift_mat = loadmat(kwargs["driftlist"])
        # convert driftlist into DataFrame
        drift_df = pd.DataFrame(
            drift_mat["driftlist"][:, :3], columns=["frame", "dy", "dx"]
        )
        drift_df = drift_df.astype({"frame": int}).set_index("frame")

        # load aoiinfo mat file
        aoi_mat = {}
        aoi_df = {}
        for dtype in dtypes:
            try:
                aoi_mat[dtype] = loadmat(kwargs[f"{dtype}-aoiinfo"])
            except ValueError:
                aoi_mat[dtype] = np.loadtxt(kwargs[f"{dtype}-aoiinfo"])
            try:
                aoi_df[dtype] = pd.DataFrame(
                    aoi_mat[dtype]["aoiinfo2"],
                    columns=["frame", "ave", "y", "x", "pixnum", "aoi"],
                )
            except KeyError:
                aoi_df[dtype] = pd.DataFrame(
                    aoi_mat[dtype]["aoifits"]["aoiinfo2"][0, 0],
                    columns=["frame", "ave", "y", "x", "pixnum", "aoi"],
                )
            except IndexError:
                aoi_df[dtype] = pd.DataFrame(
                    aoi_mat[dtype], columns=["frame", "ave", "y", "x", "pixnum", "aoi"]
                )
            aoi_df[dtype] = aoi_df[dtype].astype({"aoi": int}).set_index("aoi")
            # adjust to python indexing
            aoi_df[dtype]["x"] = aoi_df[dtype]["x"] - 1
            aoi_df[dtype]["y"] = aoi_df[dtype]["y"] - 1

        # calculate the cumulative sum of dx and dy relative to the aoiinfo frame
        aoiinfo_frame = int(aoi_df["ontarget"].at[1, "frame"])
        drift_df.loc[aoiinfo_frame + 1 :] = (
            drift_df.loc[aoiinfo_frame + 1 :].cumsum(axis=0).values
        )
        drift_df.loc[aoiinfo_frame - 1 :: -1] = (
            (-drift_df.loc[aoiinfo_frame : drift_df.index[1] : -1])
            .cumsum(axis=0)
            .values
        )

        if ("frame-start" in kwargs) and ("frame-end" in kwargs):
            f1 = int(kwargs["frame-start"])
            f2 = int(kwargs["frame-end"])
            drift_df = drift_df.loc[f1:f2]

        labels = defaultdict(lambda: None)
        for dtype in dtypes:
            if f"{dtype}-labels" in kwargs:
                labels_mat = loadmat(kwargs[f"{dtype}-labels"])
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
        self.config = kwargs
        self.header = header
        self.dtypes = dtypes
        self.aoiinfo = aoi_df
        self.cumdrift = drift_df
        self.labels = labels
        self.name = kwargs["name"]

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
        glimpse_path = Path(self.config["glimpse-folder"]) / f"{glimpse_number}.glimpse"
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


def read_glimpse(path, **kwargs):
    """
    Preprocess glimpse files.
    """
    P = kwargs.pop("P")
    C = kwargs.pop("num-channels")
    name = kwargs.pop("dataset")
    channels = kwargs.pop("channels")

    offsets = defaultdict(int)
    # iterate over channels
    data = defaultdict(list)
    target_xy = defaultdict(list)
    for c in range(C):
        glimpse = GlimpseDataset(**kwargs, **channels[c])

        raw_target_xy = {}
        colors = {}
        colors["ontarget"] = "#AA3377"
        colors["offtarget"] = "#CCBB44"
        for dtype in glimpse.dtypes:
            N = len(glimpse.aoiinfo[dtype])
            F = len(glimpse.cumdrift)
            raw_target_xy[dtype] = (
                np.expand_dims(glimpse.aoiinfo[dtype][["x", "y"]].values, axis=1)
                + glimpse.cumdrift[["dx", "dy"]].values
            )
            target_xy[dtype].append(np.zeros((N, F, 2)))
            data[dtype].append(
                np.zeros(
                    (N, F, P, P),
                    dtype="int",
                )
            )

            # plot AOIs in raw FOV images
            fig = plt.figure(figsize=(10, 10 * glimpse.height / glimpse.width))
            ax = fig.add_subplot(1, 1, 1)
            fov = glimpse[glimpse.cumdrift.index[0]]
            vmin = np.percentile(fov, 1)
            vmax = np.percentile(fov, 99)
            ax.imshow(fov, vmin=vmin, vmax=vmax, cmap="gray")
            for aoi in glimpse.aoiinfo[dtype].index:
                # areas of interest
                y_pos = round(glimpse.aoiinfo[dtype].at[aoi, "y"] - 0.5 * (P - 1)) - 0.5
                x_pos = round(glimpse.aoiinfo[dtype].at[aoi, "x"] - 0.5 * (P - 1)) - 0.5
                ax.add_patch(
                    Rectangle(
                        (x_pos, y_pos),
                        P,
                        P,
                        edgecolor=colors[dtype],
                        lw=1,
                        facecolor="none",
                    )
                )
            ax.set_title(f"{dtype} AOI locations for channel {c}", fontsize=16)
            ax.set_xlabel("x", fontsize=16)
            ax.set_ylabel("y", fontsize=16)
            plt.savefig(path / f"{dtype}-channel{c}.png", dpi=300)

        # plot offset in raw FOV images
        fig = plt.figure(figsize=(10, 10 * glimpse.height / glimpse.width))
        ax = fig.add_subplot(1, 1, 1)
        fov = glimpse[glimpse.cumdrift.index[0]]
        vmin = np.percentile(fov, 1)
        vmax = np.percentile(fov, 99)
        ax.imshow(fov, vmin=vmin, vmax=vmax, cmap="gray")
        ax.add_patch(
            Rectangle(
                (10, 10),
                30,
                30,
                edgecolor="#CCBB44",
                lw=1,
                facecolor="none",
            )
        )
        ax.set_title(f"offset location for channel {c}", fontsize=16)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("y", fontsize=16)
        plt.savefig(path / f"offset-channel{c}.png", dpi=300)

        # loop through each frame
        logger.info("Processing glimpse files ...")
        for f, frame in enumerate(tqdm(glimpse.cumdrift.index)):
            img = glimpse[frame]

            values, counts = np.unique(img[10:40, 10:40], return_counts=True)
            for value, count in zip(values, counts):
                offsets[value] += count
            for dtype in glimpse.dtypes:
                # loop through each aoi
                for n, aoi in enumerate(glimpse.aoiinfo[dtype].index):
                    shiftx = round(raw_target_xy[dtype][n, f, 0] - 0.5 * (P - 1))
                    shifty = round(raw_target_xy[dtype][n, f, 1] - 0.5 * (P - 1))
                    data[dtype][c][n, f, :, :] += img[
                        shifty : shifty + P, shiftx : shiftx + P
                    ]
                    target_xy[dtype][c][n, f, 0] = (
                        raw_target_xy[dtype][n, f, 0] - shiftx
                    )
                    target_xy[dtype][c][n, f, 1] = (
                        raw_target_xy[dtype][n, f, 1] - shifty
                    )

        # assert that target positions are within central pixel
        for dtype in glimpse.dtypes:
            assert (target_xy[dtype][c] > 0.5 * P - 1).all()
            assert (target_xy[dtype][c] < 0.5 * P).all()

    min_data = np.inf
    for dtype in data.keys():
        # concatenate channels
        data[dtype] = np.stack(data[dtype], -3)
        target_xy[dtype] = np.stack(target_xy[dtype], -2)
        min_data = min(min_data, data[dtype].min())
        # convert data to torch tensor
        data[dtype] = torch.tensor(data[dtype])
        target_xy[dtype] = torch.tensor(target_xy[dtype])

    # process offset data
    offsets = OrderedDict(sorted(offsets.items()))
    offset_samples = np.array(list(offsets.keys()))
    offset_weights = np.array(list(offsets.values()))
    # if data.min() is smaller than the smallest offset then
    # add a single point to offset samples with that value - 1
    if min_data <= offset_samples[0]:
        offset_samples = np.insert(offset_samples, 0, min_data - 1)
        offset_weights = np.insert(offset_weights, 0, 1)
    # normalize weights
    offset_weights = offset_weights / offset_weights.sum()
    # convert data to torch tensor
    offset_samples = torch.tensor(offset_samples, dtype=torch.int)
    offset_weights = torch.tensor(offset_weights)

    # plot offset distribution
    plt.figure(figsize=(3, 3))
    plt.bar(offset_samples, offset_weights)
    plt.title("Empirical Distribution", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlabel("Intensity", fontsize=12)
    plt.tight_layout()
    plt.savefig(path / "offset-distribution.png", dpi=300)

    data = defaultdict(lambda: None, data)
    target_xy = defaultdict(lambda: None, target_xy)

    dataset = CosmosDataset(
        data["ontarget"],
        target_xy["ontarget"],
        glimpse.labels["ontarget"],
        data["offtarget"],
        target_xy["offtarget"],
        glimpse.labels["offtarget"],
        offset_samples,
        offset_weights,
        name=name,
    )
    save(dataset, path)

    logger.info(
        f"On-target data: N={dataset.ontarget.N} AOIs, "
        f"F={dataset.ontarget.F} frames, "
        f"C={dataset.ontarget.C} channels, "
        f"P={dataset.ontarget.P} pixels, "
        f"P={dataset.ontarget.P} pixels"
    )
    if dataset.offtarget.images is not None:
        logger.info(
            f"Off-target data: N={dataset.offtarget.N} AOIs, "
            f"F={dataset.offtarget.F} frames, "
            f"C={dataset.offtarget.C} channels, "
            f"P={dataset.offtarget.P} pixels, "
            f"P={dataset.offtarget.P} pixels"
        )
    logger.info(f"Data is saved in {Path(path) / 'data.tpqr'}")
