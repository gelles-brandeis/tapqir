# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from scipy.io import loadmat

from tapqir.utils.dataset import CosmosDataset, save

# logger
logger = logging.getLogger(__name__)


def bin_hist(
    samples: torch.Tensor, weights: torch.Tensor, s: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, r = divmod(len(samples) - 1, s)
    n = 1 + q + int(bool(r))
    new_samples = torch.zeros(n, dtype=torch.int)
    new_weights = torch.zeros(n)
    new_samples[0] = samples[0]
    new_weights[0] = weights[0]
    new_samples[1 : 1 + q] = samples[1 + s // 2 : 1 + q * s : s]
    for i in range(s):
        new_weights[1 : 1 + q] += weights[1 + i : 1 + q * s : s]
    if r:
        new_samples[-1] = samples[1 + q * s + r // 2]
        new_weights[-1] = weights[1 + q * s :].sum()
    return new_samples, new_weights


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
    """

    def __init__(self, c=0, **kwargs):
        dtypes = ["ontarget"]
        if kwargs["use-offtarget"]:
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
        drift_df["ttb"] = header["ttb"]

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

        if kwargs["frame-range"]:
            f1 = int(kwargs["frame-start"])
            f2 = int(kwargs["frame-end"])
            drift_df = drift_df.loc[f1:f2]

        labels = defaultdict(lambda: None)
        for dtype in dtypes:
            if kwargs["labels"] and kwargs[f"{dtype}-labels"] is not None:
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
        self.c = c
        self.offset_x = kwargs["offset-x"]
        self.offset_y = kwargs["offset-y"]

    def __len__(self) -> int:
        return self.F

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
        return img + 2**15

    @property
    def N(self) -> int:
        return len(self.aoiinfo["ontarget"])

    @property
    def Nc(self) -> int:
        if "offtarget" in self.dtypes:
            return len(self.aoiinfo["offtarget"])
        return 0

    @property
    def F(self) -> int:
        return len(self.cumdrift)

    def __repr__(self):
        return rf"{self.__class__.__name__}(N={self.N}, Nc={self.Nc}, F={self.F})"

    def __str__(self):
        return f"{self.__class__.__name__}(N={self.N}, Nc={self.Nc}, F={self.F})"

    def plot(
        self,
        dtypes: Tuple[str, ...],
        P: int,
        n: int = None,
        f: int = 0,
        save: bool = False,
        path: Path = None,
        ax=None,
        item: dict = {},
        title: str = None,
    ) -> None:
        """
        Plot AOIs in the field of view.

        :param dtype: Data type (``ontarget``, ``offtarget``, ``offset``).
        :param P: AOI size.
        :param path: Path where to save plots.
        :param save: Save plots.
        """
        colors = {}
        colors["ontarget"] = "#AA3377"
        colors["offtarget"] = "#CCBB44"

        if ax is None:
            fig = plt.figure(figsize=(10, 10 * self.height / self.width))
            ax = fig.add_subplot(1, 1, 1)

        frame = self.cumdrift.index[f]
        fov = self[frame]
        c = self.c
        if "fov" in item:
            item[f"fov_c{c}"].set_data(fov)
        else:
            vmin = np.percentile(fov, 1)
            vmax = np.percentile(fov, 99)
            item[f"fov_c{c}"] = ax.imshow(fov, vmin=vmin, vmax=vmax, cmap="gray")

        for dtype in dtypes:
            if dtype in ["ontarget", "offtarget"]:
                for i, aoi in enumerate(self.aoiinfo[dtype].index):
                    if dtype == "offtarget":
                        i = i + self.N
                    # areas of interest
                    y_pos = (
                        round(
                            self.aoiinfo[dtype].at[aoi, "y"]
                            + self.cumdrift.at[frame, "dy"]
                            - 0.5 * (P - 1)
                        )
                        - 0.5
                    )
                    x_pos = (
                        round(
                            self.aoiinfo[dtype].at[aoi, "x"]
                            + self.cumdrift.at[frame, "dx"]
                            - 0.5 * (P - 1)
                        )
                        - 0.5
                    )
                    if f"aoi_n{i}_c{c}" in item:
                        item[f"aoi_n{i}_c{c}"].set_xy((x_pos, y_pos))
                    else:
                        item[f"aoi_n{i}_c{c}"] = ax.add_patch(
                            Rectangle(
                                (x_pos, y_pos),
                                P,
                                P,
                                edgecolor=colors[dtype],
                                lw=1,
                                facecolor="none",
                            )
                        )
                    if n == i:
                        item[f"aoi_n{i}_c{c}"].set_edgecolor(f"C{2+c}")
                        item[f"aoi_n{i}_c{c}"].set(zorder=2)
            elif dtype == "offset":
                ax.add_patch(
                    Rectangle(
                        (self.offset_x, self.offset_y),
                        P,
                        P,
                        edgecolor="#CCBB44",
                        lw=1,
                        facecolor="none",
                    )
                )
        if title is None:
            title = rf"AOI ${n}$, Frame ${f}$"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$x$", fontsize=9)
        ax.set_ylabel(r"$y$", fontsize=9)
        if save:
            plt.savefig(path / f"{dtype}-channel{self.c}.png", dpi=300)


def read_glimpse(path, progress_bar, **kwargs):
    """
    Preprocess glimpse files.
    """
    P = kwargs.pop("P")
    C = kwargs.pop("num-channels")
    name = kwargs.pop("dataset")
    channels = kwargs.pop("channels")
    offset_P = kwargs.pop("offset-P")
    bin_size = kwargs.pop("bin-size")

    offsets = defaultdict(int)
    offset_medians = []
    # iterate over channels
    data = defaultdict(list)
    target_xy = defaultdict(list)
    labels = defaultdict(list)
    time1 = []
    ttb = []
    for c in range(C):
        logger.info(f"Channel #{c} ({channels[c]['name']})")
        glimpse = GlimpseDataset(**kwargs, **channels[c], c=c)

        raw_target_xy = {}
        colors = {}
        colors["ontarget"] = "#AA3377"
        colors["offtarget"] = "#CCBB44"
        time1.append(float(glimpse.header["time1"]))
        ttb.append(glimpse.cumdrift["ttb"].values)
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
            labels[dtype].append(glimpse.labels[dtype])

            title = f"{dtype} locations for channel {c}"
            glimpse.plot((dtype,), P, path=path, save=True, item={}, title=title)

        # plot offset in raw FOV images
        title = f"{dtype} locations for channel {c}"
        glimpse.plot(("offset",), offset_P, path=path, save=True, item={}, title=title)

        # loop through each frame
        for f, frame in enumerate(progress_bar(glimpse.cumdrift.index)):
            img = glimpse[frame]

            offset_img = img[
                glimpse.offset_y : glimpse.offset_y + offset_P,
                glimpse.offset_x : glimpse.offset_x + offset_P,
            ]
            offset_medians.append(np.median(offset_img))
            values, counts = np.unique(offset_img, return_counts=True)
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

    logger.info("Processing extracted AOIs ...")
    min_data = np.inf
    for dtype in data.keys():
        # concatenate color channels
        data[dtype] = np.stack(data[dtype], -3)
        target_xy[dtype] = np.stack(target_xy[dtype], -2)
        min_data = min(min_data, data[dtype].min())
        if any(label is None for label in labels[dtype]):
            labels[dtype] = None
        else:
            labels[dtype] = np.stack(labels[dtype], -1)
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
    # remove values from the upper 0.5 percentile
    high_mask = offset_weights.cumsum() > 0.995
    high_weights = offset_weights[high_mask].sum()
    offset_samples = offset_samples[~high_mask]
    offset_weights = offset_weights[~high_mask]
    offset_weights[-1] += high_weights
    # convert data to torch tensor
    offset_samples = torch.tensor(offset_samples, dtype=torch.int)
    offset_weights = torch.tensor(offset_weights)
    # thin offset histogram
    offset_samples, offset_weights = bin_hist(offset_samples, offset_weights, bin_size)

    data = defaultdict(lambda: None, data)
    target_xy = defaultdict(lambda: None, target_xy)
    # concatenate ontarget and offtarget
    is_ontarget = torch.cat(
        tuple(
            torch.full(
                target_xy[dtype].shape[:1], dtype == "ontarget", dtype=torch.bool
            )
            for dtype in glimpse.dtypes
        ),
        0,
    )
    data = torch.cat(tuple(data[dtype] for dtype in glimpse.dtypes), 0)
    target_xy = torch.cat(tuple(target_xy[dtype] for dtype in glimpse.dtypes), 0)
    time1 = torch.as_tensor(time1)
    ttb = torch.as_tensor(np.array(ttb)).T
    if all(labels[dtype] is None for dtype in glimpse.dtypes):
        labels = None
    else:
        labels = np.concatenate(
            tuple(
                labels[dtype] for dtype in glimpse.dtypes if labels[dtype] is not None
            ),
            0,
        )

    dataset = CosmosDataset(
        data,
        target_xy,
        is_ontarget,
        labels=labels,
        offset_samples=offset_samples,
        offset_weights=offset_weights,
        time1=time1,
        ttb=ttb,
        name=name,
    )
    logger.info(
        f"Dataset: N={dataset.N} on-target AOIs, "
        f"Nc={dataset.Nc} off-target AOIs, "
        f"F={dataset.F} frames, "
        f"C={dataset.C} channels, "
        f"Px={dataset.P} pixels, "
        f"Py={dataset.P} pixels"
    )
    save(dataset, path)

    logger.info("- saving images")
    # plot offset distribution
    plt.figure(figsize=(3, 3))
    plt.bar(offset_samples, offset_weights, alpha=0.5, label="Offset")
    # plot data distribution for each channel
    for c in range(C):
        data_samples, data_counts = torch.unique(
            data[:, :, c], sorted=True, return_counts=True
        )
        data_weights = data_counts / data_counts.sum()
        plt.bar(data_samples, data_weights, alpha=0.5, label=f"Channel {c}")
    plt.title("Empirical Distribution", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlabel("Intensity", fontsize=12)
    plt.xlim(offset_samples.min(), dataset.vmax.max())
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "offset-distribution.png", dpi=300)

    plt.figure(figsize=(5, 3))
    plt.plot(offset_medians, label="Offset Median")
    plt.title("Offset drift", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.xlabel("Frames", fontsize=12)
    plt.ylim(offset_samples.min(), offset_samples.max())
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "offset-medians.png", dpi=300)
