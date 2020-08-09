import os
import configparser
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from cosmos.utils.dataset import CoSMoSDataset


def read_glimpse(path, D):
    """ Read Glimpse files """

    device = torch.device("cpu")
    # read options.cfg fiel
    config = configparser.ConfigParser(allow_no_value=True)
    cfg_file = os.path.join(path, "options.cfg")
    config.read(cfg_file)

    dtypes = ["test"]
    if config["glimpse"]["control_aoiinfo"] is not None:
        dtypes.append("control")

    # convert header into dict format
    mat_header = loadmat(os.path.join(config["glimpse"]["dir"], "header.mat"))
    header = dict()
    for i, dt in enumerate(mat_header["vid"].dtype.names):
        header[dt] = np.squeeze(mat_header["vid"][0, 0][i])

    # load driftlist mat file
    drift_mat = loadmat(config["glimpse"]["driftlist"])
    # calculate the cumulative sum of dx and dy
    drift_mat["driftlist"][:, 1:3] = np.cumsum(
        drift_mat["driftlist"][:, 1:3], axis=0)
    # convert driftlist into DataFrame
    drift_df = pd.DataFrame(
        drift_mat["driftlist"][:, :3],
        columns=["frame", "dx", "dy"])
    drift_df = drift_df.astype({"frame": int}).set_index("frame")

    # load aoiinfo mat file
    aoi_mat = {}
    aoi_df = {}
    for dtype in dtypes:
        aoi_mat[dtype] = loadmat(config["glimpse"]["{}_aoiinfo".format(dtype)])
        aoi_df[dtype] = pd.DataFrame(
            aoi_mat[dtype]["aoiinfo2"],
            columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
        aoi_df[dtype] = aoi_df[dtype].astype({"aoi": int}).set_index("aoi")
        aoi_df[dtype]["x"] = aoi_df[dtype]["x"] \
            - drift_df.at[int(aoi_df[dtype].at[1, "frame"]), "dx"]
        aoi_df[dtype]["y"] = aoi_df[dtype]["y"] \
            - drift_df.at[int(aoi_df[dtype].at[1, "frame"]), "dy"]

    labels = None
    if config["glimpse"]["labels"]:
        if config["glimpse"]["labeltype"] == "framelist":
            framelist = loadmat(config["glimpse"]["labels"])
            f1 = framelist[name.split("-")[0].split(".")[0]][0, 2]
            f2 = framelist[name.split("-")[0].split(".")[0]][-1, 2]
            drift_df = drift_df.loc[f1:f2]
            aoi_list = np.unique(framelist[name.split("-")[0].split(".")[0]][:, 0])
            aoi_df = aoi_df.loc[aoi_list]
            labels = np.zeros(
                (len(aoi_df), len(drift_df)),
                dtype=[("aoi", int), ("frame", int), ("z", int), ("spotpicker", int)])
            labels["aoi"] = framelist[name.split("-")[0].split(".")[0]][:, 0].reshape(len(aoi_df), len(drift_df))
            labels["frame"] = framelist[name.split("-")[0].split(".")[0]][:, 2].reshape(len(aoi_df), len(drift_df))
            labels["spotpicker"] = framelist[name.split("-")[0].split(".")[0]][:, 1] \
                .reshape(len(aoi_df), len(drift_df))
            labels["spotpicker"][labels["spotpicker"] == 0] = 3
            labels["spotpicker"][labels["spotpicker"] == 2] = 0
        else:
            labels_mat = loadmat(config["glimpse"]["labels"])
            labels = np.zeros(
                (len(aoi_df["test"]), len(drift_df)),
                dtype=[("aoi", int), ("frame", int), ("z", bool), ("spotpicker", float)])
            labels["aoi"] = aoi_df["test"].index.values.reshape(-1, 1)
            labels["frame"] = drift_df.index.values
            spot_picker = labels_mat["Intervals"][
                "CumulativeIntervalArray"][0, 0]
            for sp in spot_picker:
                aoi = int(sp[-1])
                start = int(sp[1])
                end = int(sp[2])
                if sp[0] in [-2., 0., 2.]:
                    labels["spotpicker"][(labels["aoi"] == aoi) &
                                         (labels["frame"] >= start) &
                                         (labels["frame"] <= end)] = 0
                elif sp[0] in [-3., 1., 3.]:
                    labels["spotpicker"][(labels["aoi"] == aoi) &
                                         (labels["frame"] >= start) &
                                         (labels["frame"] <= end)] = 1

        labels["z"] = labels["spotpicker"]

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
        data={"dx": 0., "dy": 0., "abs_dx": drift_df["dx"],
              "abs_dy": drift_df["dy"]},
        index=drift_df.index)

    target = {}
    data = {}
    for dtype in dtypes:
        # target location
        target[dtype] = pd.DataFrame(
            data={"frame": aoi_df[dtype]["frame"], "x": 0., "y": 0.,
                  "abs_x": aoi_df[dtype]["x"], "abs_y": aoi_df[dtype]["y"]},
            index=aoi_df[dtype].index)
        data[dtype] = np.ones((len(aoi_df[dtype]), len(drift_df), D, D)) * 2**15
    offsets = np.ones((len(drift), 4, 30, 30)) * 2**15
    # loop through each frame
    for i, frame in enumerate(tqdm(drift.index)):
        # read the entire frame image
        glimpse_number = header["filenumber"][frame - 1]
        glimpse_path = os.path.join(
            config["glimpse"]["dir"], "{}.glimpse".format(glimpse_number))
        offset = header["offset"][frame - 1]
        with open(glimpse_path, "rb") as fid:
            fid.seek(offset)
            img = np.fromfile(
                fid, dtype='>i2',
                count=height*width) \
                .reshape(height, width)

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
                top_x = int((aoi_df[dtype].at[aoi, "x"] - D * 0.5) // 1
                            + drift_df.at[frame, "dx"] // 1)
                left_y = int((aoi_df[dtype].at[aoi, "y"] - D * 0.5) // 1
                             + drift_df.at[frame, "dy"] // 1)
                # j-th frame, i-th aoi
                data[dtype][j, i, :, :] += img[top_x:top_x+D, left_y:left_y+D]
                # new target center for the first frame
                # if i == 0:
                #    target.at[aoi, "x"] = aoi_df.at[aoi, "x"] - top_x - 1
                #    target.at[aoi, "y"] = aoi_df.at[aoi, "y"] - left_y - 1
    offset = torch.tensor(offsets, dtype=torch.float32)
    for dtype in dtypes:
        for j, aoi in enumerate(aoi_df[dtype].index):
            target[dtype].at[aoi, "x"] = aoi_df[dtype].at[aoi, "x"] \
                - int((aoi_df[dtype].at[aoi, "x"] - D * 0.5) // 1) - 1
            target[dtype].at[aoi, "y"] = aoi_df[dtype].at[aoi, "y"] \
                - int((aoi_df[dtype].at[aoi, "y"] - D * 0.5) // 1) - 1
        # convert data into torch tensor
        data[dtype] = torch.tensor(data[dtype], dtype=torch.float32)
        dataset = CoSMoSDataset(data[dtype], target[dtype], drift, dtype, device, offset, labels)
        dataset.save(path)
