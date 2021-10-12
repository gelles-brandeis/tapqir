# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyro.ops.stats import quantile
from scipy.io import loadmat
from tqdm import tqdm

from tapqir.utils.dataset import CosmosDataset, save

# logger
logger = logging.getLogger(__name__)


class GlimpseDataset:
    """
    GlimpseDataset parses header, aoiinfo, driftlist, and intervals (optional)
    files and creates

    1. aoiinfo and cumdrift DataFrames
    2. __getitem__ method to retrieve glimpse image for a given frame
    3. labels np.array

        title, header_dir, ontarget_aoiinfo, offtarget_aoiinfo
        driftlist, frame_start, frame_end, ontarget_labels, offtarget_labels
    :param title: Project/experiment name.
    :param header_dir: Path to the header/glimpse folder.
    :param ontarget_aoiinfo: Path to the on-target AOI locations file.
    :param offtarget_aoiinfo: Path to the off-target control AOI locations file (optional).
    :param driftlist: Path to the driftlist file.
    :param frame_start: First frame to include in the analysis (optional).
    :param frame_end: Last frame to include in the analysis (optional).
    :param ontarget_labels: Path to the on-target label intervals file.
    :param offtarget_labels: Path to the off-target label intervals file.
    """

    def __init__(self, **kwargs):
        dtypes = ["ontarget"]
        if kwargs["offtarget_aoiinfo"] is not None:
            dtypes.append("offtarget")

        # convert header into dict format
        mat_header = loadmat(Path(kwargs["header_dir"]) / "header.mat")
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
                aoi_mat[dtype] = loadmat(kwargs[f"{dtype}_aoiinfo"])
            except ValueError:
                aoi_mat[dtype] = np.loadtxt(kwargs[f"{dtype}_aoiinfo"])
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

        if kwargs["frame_start"] and kwargs["frame_end"]:
            f1 = int(kwargs["frame_start"])
            f2 = int(kwargs["frame_end"])
            drift_df = drift_df.loc[f1:f2]

        labels = defaultdict(lambda: None)
        for dtype in dtypes:
            if kwargs[f"{dtype}_labels"] is not None:
                labels_mat = loadmat(kwargs[f"{dtype}_labels"])
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
        self.title = kwargs["title"]

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
        glimpse_path = Path(self.config["header_dir"]) / f"{glimpse_number}.glimpse"
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
    P = int(kwargs.pop("aoi_size"))
    glimpse = GlimpseDataset(**kwargs)

    raw_target_xy = defaultdict(lambda: None)
    target_xy = defaultdict(lambda: None)
    data = defaultdict(lambda: None)
    for dtype in glimpse.dtypes:
        raw_target_xy[dtype] = (
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
    logger.info("Processing glimpse files ...")
    for f, frame in enumerate(tqdm(glimpse.cumdrift.index)):
        img = glimpse[frame]

        offsets[f, 0, :, :] += img[10:40, 10:40]
        offsets[f, 1, :, :] += img[10:40, -40:-10]
        offsets[f, 2, :, :] += img[-40:-10, 10:40]
        offsets[f, 3, :, :] += img[-40:-10, -40:-10]
        for dtype in glimpse.dtypes:
            # loop through each aoi
            for n, aoi in enumerate(glimpse.aoiinfo[dtype].index):
                shiftx = round(raw_target_xy[dtype][n, f, 0] - 0.5 * (P - 1))
                shifty = round(raw_target_xy[dtype][n, f, 1] - 0.5 * (P - 1))
                data[dtype][n, f, :, :] += img[shifty : shifty + P, shiftx : shiftx + P]
                target_xy[dtype][n, f, 0] = raw_target_xy[dtype][n, f, 0] - shiftx
                target_xy[dtype][n, f, 1] = raw_target_xy[dtype][n, f, 1] - shifty
    offset = torch.tensor(offsets)
    for dtype in glimpse.dtypes:
        assert (target_xy[dtype] > 0.5 * P - 1).all()
        assert (target_xy[dtype] < 0.5 * P).all()
        # convert data into torch tensor
        data[dtype] = torch.tensor(data[dtype])
        target_xy[dtype] = torch.tensor(target_xy[dtype])

    # process offset data
    offset_min = quantile(offset.flatten().float(), 0.005).item()
    offset_max = quantile(offset.flatten().float(), 0.995).item()
    clamped_offset = torch.clamp(offset, offset_min, offset_max)
    offset_samples, offset_weights = torch.unique(
        clamped_offset, sorted=True, return_counts=True
    )
    offset_weights = offset_weights.float() / offset_weights.sum()

    dataset = CosmosDataset(
        data["ontarget"],
        target_xy["ontarget"],
        glimpse.labels["ontarget"],
        data["offtarget"],
        target_xy["offtarget"],
        glimpse.labels["offtarget"],
        offset_samples,
        offset_weights,
        title=glimpse.title,
    )
    save(dataset, path)

    logger.info(
        f"On-target data: N={dataset.ontarget.N} AOIs, "
        f"F={dataset.ontarget.F} frames, "
        f"P={dataset.ontarget.P} pixels, "
        f"P={dataset.ontarget.P} pixels"
    )
    if dataset.offtarget.images is not None:
        logger.info(
            f"Off-target data: N={dataset.offtarget.N} AOIs, "
            f"F={dataset.offtarget.F} frames, "
            f"P={dataset.offtarget.P} pixels, "
            f"P={dataset.offtarget.P} pixels"
        )
    logger.info(f"Data is saved in {Path(path) / 'data.tpqr'}")
