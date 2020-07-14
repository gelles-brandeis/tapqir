import os
import configparser
from scipy.io import loadmat
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pyro


class CoSMoSDataset(Dataset):
    """ CoSMoS Dataset """

    def __init__(self, data=None, target=None, drift=None, labels=None, dtype=None, device=None, offset=None):
        self.data = data.to(device)
        self.N, self.F, self.D, _ = self.data.shape
        self.target = target
        self.drift = drift
        if dtype == "test":
            self.labels = labels
            self.offset = offset
            self.offset_median = torch.median(self.offset)
            self.offset_mean = torch.mean(self.offset)
            self.offset_var = torch.var(self.offset)
            self.data_median = torch.median(self.data)
            self.noise = (self.data.std(dim=(1, 2, 3)).mean() - self.offset.std()) * np.pi * (2 * 1.3) ** 2
            offset_max = np.percentile(self.offset.cpu().numpy(), 99.5)
            offset_min = np.percentile(self.offset.cpu().numpy(), 0.5)
            offset_weights, offset_samples = np.histogram(self.offset.cpu().numpy(),
                                                 range=(offset_min, offset_max),
                                                 bins=max(1, int(offset_max - offset_min)),
                                                 #bins=8,
                                                 density=True)
            offset_samples = offset_samples[:-1]
            self.offset_samples = torch.from_numpy(offset_samples).float().to(device)
            self.offset_weights = torch.from_numpy(offset_weights).float().to(device)
        assert self.N == len(self.target)
        assert self.F == len(self.drift)
        self.dtype = dtype
        self.device = device
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
            np.save(os.path.join(path, "labels.npy"),
                    self.labels)
            torch.save(self.offset, os.path.join(
                path, "offset.pt"))

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

def read_glimpse(name, D, dtype, device=None):
    """ Read Glimpse files """

    """
    read header, aoiinfo, driftlist, and labels files
    """
    config = configparser.ConfigParser(allow_no_value=True)
    config.read("datasets.cfg")
    files = ["dir", "header", "test_aoiinfo", "control_aoiinfo", "driftlist", "labels"]
    path_to = {}
    if name.split(".")[0] in config:
        for FILE in files:
            path_to[FILE] = config[name.split(".")[0]][FILE]
    else:
        config.add_section(name)
        for FILE in files:
            path_to[FILE] = input("{}: ".format(FILE))
            config.set(name, FILE, path_to[FILE])
        with open("datasets.cfg", "w") as configfile:
            config.write(configfile)

    """
    convert header.mat into dict format
    convert driftlist.mat into DataFrame format
        and calculate cumulative sum of the drift across frames
    convert aoiinfo.mat into DataFrame and
        adjust target position to frame #1
    convert labels.mat into MutliIndex DataFrame
    select subset of frames and aois
    """
    # convert header into dict format
    mat_header = loadmat(path_to["header"])
    header = dict()
    for i, dt in enumerate(mat_header["vid"].dtype.names):
        header[dt] = np.squeeze(mat_header["vid"][0, 0][i])

    # load driftlist mat file
    drift_mat = loadmat(path_to["driftlist"])
    # calculate the cumulative sum of dx and dy
    drift_mat["driftlist"][:, 1:3] = np.cumsum(
        drift_mat["driftlist"][:, 1:3], axis=0)
    # convert driftlist into DataFrame
    drift_df = pd.DataFrame(
        drift_mat["driftlist"][:, :3],
        columns=["frame", "dx", "dy"])
    drift_df = drift_df.astype({"frame": int}).set_index("frame")

    # load aoiinfo mat file
    aoi_mat = loadmat(path_to["{}_aoiinfo".format(dtype)])
    # convert aoiinfo into DataFrame
    if name in ["Gracecy3"]:
        aoi_df = pd.DataFrame(
            aoi_mat["aoifits"]["aoiinfo2"][0, 0],
            columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
    else:
        aoi_df = pd.DataFrame(
            aoi_mat["aoiinfo2"],
            columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
    aoi_df = aoi_df.astype({"aoi": int}).set_index("aoi")
    aoi_df["x"] = aoi_df["x"] \
        - drift_df.at[int(aoi_df.at[1, "frame"]), "dx"]
    aoi_df["y"] = aoi_df["y"] \
        - drift_df.at[int(aoi_df.at[1, "frame"]), "dy"]

    if name in ["LarryCy3sigma54Short",
                     "LarryCy3sigma54NegativeControlShort"]:
        f1 = 170
        f2 = 1000
        drift_df = drift_df.loc[f1:f2]
        aoi_list = np.arange(1, 33)
        aoi_df = aoi_df.loc[aoi_list]
    elif name in ["LarryCy3sigma54NegativeControl1"]:
        aoi_df = aoi_df.loc[:64]
    elif name in ["LarryCy3sigma54NegativeControl2"]:
        aoi_df = aoi_df.loc[65:]
    elif name in ["GraceArticlePol2NegativeControl1"]:
        aoi_df = aoi_df.loc[:263]
    elif name in ["GraceArticlePol2NegativeControl2"]:
        aoi_df = aoi_df.loc[264:]

    labels = None
    if path_to["labels"]:
        if name.startswith("FL"):
            framelist = loadmat(path_to["labels"])
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
            labels_mat = loadmat(path_to["labels"])
            labels = np.zeros(
                (len(aoi_df), len(drift_df)),
                dtype=[("aoi", int), ("frame", int), ("z", bool), ("spotpicker", float)])
            labels["aoi"] = aoi_df.index.values.reshape(-1, 1)
            labels["frame"] = drift_df.index.values
            spot_picker = labels_mat["Intervals"][
                "CumulativeIntervalArray"][0, 0]
            for sp in spot_picker:
                aoi = int(sp[-1])
                start = int(sp[1])
                end = int(sp[2])
                if sp[0] in [-2., 0., 2.]:
                    labels["spotpicker"][(labels["aoi"] == aoi) & \
                                         (labels["frame"] >= start) & \
                                         (labels["frame"] <= end)] = 0
                elif sp[0] in [-3., 1., 3.]:
                    labels["spotpicker"][(labels["aoi"] == aoi) & \
                                         (labels["frame"] >= start) & \
                                         (labels["frame"] <= end)] = 1

        labels["z"] = labels["spotpicker"]

    if name.endswith(".clean"):
        pyro.clear_param_store()
        pyro.get_param_store().load(
            filename=os.path.join("{}/runs/marginalv0.9.9/nojit/lr0.005/Adam/32".format(name.split("-")[0].split(".")[0]), "params"),
            map_location=torch.device("cpu"))
        data_stat = load_data(name.split("-")[0].split(".")[0], dtype="test", device=torch.device("cpu"))
        control_stat = load_data(name.split("-")[0].split(".")[0], dtype="control", device=torch.device("cpu"))
        bg_stat = data_stat.data.mean(dim=(1,2,3))
        try:
            bg_fit = (pyro.param("d/background_loc") + data_stat.offset_median).data.reshape(-1)
        except:
            bg_fit = (pyro.param("c/background_loc") + data_stat.offset_median).data.reshape(-1)
        fs = data_stat.target.index[torch.abs(bg_stat - bg_fit) < 8]
        aoi_df = aoi_df.loc[fs]
        labels = labels[torch.abs(bg_stat - bg_fit) < 8]

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
    N = len(aoi_df)
    F = len(drift_df)
    # target location
    target = pd.DataFrame(
        data={"frame": aoi_df["frame"], "x": 0., "y": 0.,
              "abs_x": aoi_df["x"], "abs_y": aoi_df["y"]},
        index=aoi_df.index)
    # drift
    drift = pd.DataFrame(
        data={"dx": 0., "dy": 0., "abs_dx": drift_df["dx"],
              "abs_dy": drift_df["dy"]},
        index=drift_df.index)
    data = np.ones((N, F, D, D)) * 2**15
    offsets = np.ones((F, 4, 30, 30)) * 2**15
    # loop through each frame
    for i, frame in enumerate(tqdm(drift_df.index)):
        # read the entire frame image
        glimpse_number = header["filenumber"][frame - 1]
        glimpse_path = os.path.join(
            path_to["dir"], "{}.glimpse".format(glimpse_number))
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
        # loop through each aoi
        for j, aoi in enumerate(aoi_df.index):
            # top left corner of aoi
            # integer part (target center - half aoi width) \
            # + integer part (drift)
            top_x = int((aoi_df.at[aoi, "x"] - D * 0.5) // 1
                        + drift_df.at[frame, "dx"] // 1)
            left_y = int((aoi_df.at[aoi, "y"] - D * 0.5) // 1
                         + drift_df.at[frame, "dy"] // 1)
            # j-th frame, i-th aoi
            data[j, i, :, :] += img[top_x:top_x+D, left_y:left_y+D]
            # new target center for the first frame
            #if i == 0:
            #    target.at[aoi, "x"] = aoi_df.at[aoi, "x"] - top_x - 1
            #    target.at[aoi, "y"] = aoi_df.at[aoi, "y"] - left_y - 1
    for j, aoi in enumerate(aoi_df.index):
        target.at[aoi, "x"] = aoi_df.at[aoi, "x"] \
            - int((aoi_df.at[aoi, "x"] - D * 0.5) // 1) - 1
        target.at[aoi, "y"] = aoi_df.at[aoi, "y"] \
            - int((aoi_df.at[aoi, "y"] - D * 0.5) // 1) - 1
    # convert data into torch tensor
    data = torch.tensor(data, dtype=torch.float32, device=device)
    offset = torch.tensor(offsets, dtype=torch.float32, device=device)

    return CoSMoSDataset(data, target, drift, labels, dtype, device, offset)
