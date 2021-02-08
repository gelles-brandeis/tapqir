import configparser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tqdm import tqdm

from tapqir.utils.dataset import CosmosDataset


def read_glimpse(path, D):
    """ Read Glimpse files """

    path = Path(path)
    device = torch.device("cpu")
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
    drift_mat["driftlist"][:, 1:3] = np.cumsum(drift_mat["driftlist"][:, 1:3], axis=0)
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
            aoi_df[dtype]["x"] - drift_df.at[int(aoi_df[dtype].at[1, "frame"]), "dx"]
        )
        aoi_df[dtype]["y"] = (
            aoi_df[dtype]["y"] - drift_df.at[int(aoi_df[dtype].at[1, "frame"]), "dy"]
        )

    labels = defaultdict(None)
    for dtype in dtypes:
        if config["glimpse"][f"{dtype}_labels"] is not None:
            if config["glimpse"]["labeltype"] is not None:
                framelist = loadmat(config["glimpse"][f"{dtype}_labels"])
                f1 = framelist[config["glimpse"]["labeltype"]][0, 2]
                f2 = framelist[config["glimpse"]["labeltype"]][-1, 2]
                drift_df = drift_df.loc[f1:f2]
                aoi_list = np.unique(framelist[config["glimpse"]["labeltype"]][:, 0])
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

    """
    target DataFrame
    aoi frame x y abs_x abs_y
    drfit DataFrame
    frame dx dy abs_dx abs_dy
    dx = dx % 1
    top_x = int((x - D * 0.5) // 1 + dx // 1)
    x = x - top_x - 1
    """
    height, width = int(header["height"]), int(header["width"])
    # drift
    drift = pd.DataFrame(
        data={"dx": 0.0, "dy": 0.0, "abs_dx": drift_df["dx"], "abs_dy": drift_df["dy"]},
        index=drift_df.index,
    )

    target = {}
    data = {}
    for dtype in dtypes:
        # target location
        target[dtype] = pd.DataFrame(
            data={
                "frame": aoi_df[dtype]["frame"],
                "x": 0.0,
                "y": 0.0,
                "abs_x": aoi_df[dtype]["x"],
                "abs_y": aoi_df[dtype]["y"],
            },
            index=aoi_df[dtype].index,
        )
        data[dtype] = np.ones((len(aoi_df[dtype]), len(drift_df), D, D)) * 2 ** 15
    offsets = np.ones((len(drift), 4, 30, 30)) * 2 ** 15
    # loop through each frame
    for i, frame in enumerate(tqdm(drift.index)):
        # read the entire frame image
        glimpse_number = header["filenumber"][frame - 1]
        glimpse_path = Path(config["glimpse"]["dir"]) / f"{glimpse_number}.glimpse"
        offset = header["offset"][frame - 1]
        with open(glimpse_path, "rb") as fid:
            fid.seek(offset)
            img = np.fromfile(fid, dtype=">i2", count=height * width).reshape(
                height, width
            )

        offsets[i, 0, :, :] += img[10:40, 10:40]
        offsets[i, 1, :, :] += img[10:40, -40:-10]
        offsets[i, 2, :, :] += img[-40:-10, 10:40]
        offsets[i, 3, :, :] += img[-40:-10, -40:-10]
        # new drift list (fractional part)
        drift.at[frame, "dx"] = drift_df.at[frame, "dx"] % 1
        drift.at[frame, "dy"] = drift_df.at[frame, "dy"] % 1
        for dtype in dtypes:
            # loop through each aoi
            for j, aoi in enumerate(aoi_df[dtype].index):
                # top left corner of aoi
                # integer part (target center - half aoi width) \
                # + integer part (drift)
                top_x = int(
                    (aoi_df[dtype].at[aoi, "x"] - D * 0.5) // 1
                    + drift_df.at[frame, "dx"] // 1
                )
                left_y = int(
                    (aoi_df[dtype].at[aoi, "y"] - D * 0.5) // 1
                    + drift_df.at[frame, "dy"] // 1
                )
                # j-th frame, i-th aoi
                data[dtype][j, i, :, :] += img[top_x : top_x + D, left_y : left_y + D]
                # new target center for the first frame
                # if i == 0:
                #    target.at[aoi, "x"] = aoi_df.at[aoi, "x"] - top_x - 1
                #    target.at[aoi, "y"] = aoi_df.at[aoi, "y"] - left_y - 1
    offset = torch.tensor(offsets, dtype=torch.float32)
    for dtype in dtypes:
        for j, aoi in enumerate(aoi_df[dtype].index):
            target[dtype].at[aoi, "x"] = (
                aoi_df[dtype].at[aoi, "x"]
                - int((aoi_df[dtype].at[aoi, "x"] - D * 0.5) // 1)
                - 1
            )
            target[dtype].at[aoi, "y"] = (
                aoi_df[dtype].at[aoi, "y"]
                - int((aoi_df[dtype].at[aoi, "y"] - D * 0.5) // 1)
                - 1
            )
        # convert data into torch tensor
        data[dtype] = torch.tensor(data[dtype], dtype=torch.float32)
        dataset = CosmosDataset(
            data[dtype], target[dtype], drift, dtype, device, labels[dtype], offset
        )
        dataset.save(path)
