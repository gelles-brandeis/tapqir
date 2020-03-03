import os
import configparser
from scipy.io import loadmat
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Sampler(Dataset):
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return idx


class GlimpseDataset(Dataset):
    """ CoSMoS Dataset """

    def __init__(self, dataset=None, experiment=None, device=None, path=None):
        # create logger
        self.logger = logging.getLogger(__name__)
        # store metadata
        self.name = dataset
        self.xp = experiment
        if path is not None:
            self.path = path
        else:
            self.path = os.path.join(os.getcwd(), self.name)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        fh = logging.FileHandler(os.path.join(
            self.path, "load_data.log"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
        self.load_data()
        #self.script()

    def load_data(self):
        try:
            self._store = torch.load(os.path.join(
                self.path, "{}.pt".format(self.xp)),
                map_location=self.device).detach()
            self.N, self.F, self.D, _ = self._store.shape
            self.vmin = np.percentile(self._store.cpu().numpy(), 5)
            self.vmax = np.percentile(self._store.cpu().numpy(), 99)
            # self._store = self._store.cpu()
            # assert (self.N, self.F, self.D, self.D) == self._store.shape
            self.target = pd.read_csv(os.path.join(
                self.path, "{}_target.csv".format(self.xp)),
                index_col="aoi")
            self.drift = pd.read_csv(os.path.join(
                self.path, "{}_drift.csv".format(self.xp)),
                index_col="frame")
            if os.path.isfile(os.path.join(self.path, "{}_labels.csv".format(self.xp))):
                self.labels = pd.read_csv(os.path.join(
                    self.path, "{}_labels.csv".format(self.xp)),
                    index_col=["aoi", "frame"])
            #self.logger.debug(
            #    "Loaded data from data.pt, \
            #     target.csv, and drift.csv files")
        except:
            self.read_cfg()
            self.read_mat()
            self.read_glimpse()

    def read_cfg(self):
        """
        read header, aoiinfo, driftlist, and labels files
        """
        config = configparser.ConfigParser(allow_no_value=True)
        config.read("datasets.cfg")
        files = ["dir", "header", "aoiinfo", "driftlist", "labels"]
        self.path_to = {}
        if self.name in config:
            for FILE in files:
                self.path_to[FILE] = config[self.name][FILE]
            self.logger.debug("Read filenames from dataset.cfg")
        else:
            self.logger.debug("Input filenames")
            config.add_section(self.name)
            for FILE in files:
                self.path_to[FILE] = input("{}: ".format(FILE))
                config.set(self.name, FILE, self.path_to[FILE])
            with open("datasets.cfg", "w") as configfile:
                config.write(configfile)

    def read_mat(self):
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
        self.logger.debug("Reading header.mat file ... ")
        mat_header = loadmat(self.path_to["header"])
        self.header = dict()
        for i, dt in enumerate(mat_header["vid"].dtype.names):
            self.header[dt] = np.squeeze(mat_header["vid"][0, 0][i])

        # load driftlist mat file
        self.logger.debug(
            "Reading {} file ... ".format(self.path_to["driftlist"]))
        drift_mat = loadmat(self.path_to["driftlist"])
        # calculate the cumulative sum of dx and dy
        self.logger.debug("calculating cumulative drift ... ")
        drift_mat["driftlist"][:, 1:3] = np.cumsum(
            drift_mat["driftlist"][:, 1:3], axis=0)
        # convert driftlist into DataFrame
        self.drift_df = pd.DataFrame(
            drift_mat["driftlist"][:, :3],
            columns=["frame", "dx", "dy"])
        self.drift_df = self.drift_df.astype({"frame": int}).set_index("frame")

        # load aoiinfo mat file
        self.logger.debug(
            "Reading {} file ... ".format(self.path_to["aoiinfo"]))
        aoi_mat = loadmat(self.path_to["aoiinfo"])
        # convert aoiinfo into DataFrame
        if self.name in ["Gracecy3", "Gracecy3Supervised"]:
            self.aoi_df = pd.DataFrame(
                aoi_mat["aoifits"]["aoiinfo2"][0, 0],
                columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
        else:
            self.aoi_df = pd.DataFrame(
                aoi_mat["aoiinfo2"],
                columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
        self.aoi_df = self.aoi_df.astype({"aoi": int}).set_index("aoi")
        self.aoi_df["x"] = self.aoi_df["x"] \
            - self.drift_df.at[int(self.aoi_df.at[1, "frame"]), "dx"]
        self.aoi_df["y"] = self.aoi_df["y"] \
            - self.drift_df.at[int(self.aoi_df.at[1, "frame"]), "dy"]
        self.logger.debug(
            "Adjusted target position from frame {} to frame 1 ... "
            .format(self.aoi_df.at[1, "frame"]))

        if self.name in ["LarryCy3sigma54Short",
                         "LarryCy3sigma54NegativeControlShort"]:
            f1 = 170
            f2 = 1000
            self.drift_df = self.drift_df.loc[f1:f2]
            aoi_list = np.arange(1, 33)
            self.aoi_df = self.aoi_df.loc[aoi_list]
        elif self.name in ["LarryCy3sigma54NegativeControl1"]:
            self.aoi_df = self.aoi_df.loc[:64]
        elif self.name in ["LarryCy3sigma54NegativeControl2"]:
            self.aoi_df = self.aoi_df.loc[65:]
        elif self.name in ["GraceArticlePol2NegativeControl1"]:
            self.aoi_df = self.aoi_df.loc[:263]
        elif self.name in ["GraceArticlePol2NegativeControl2"]:
            self.aoi_df = self.aoi_df.loc[264:]
        elif self.name in ["Gracecy3Short"]:
            aoi_list = np.arange(160, 240)
            self.aoi_df = self.aoi_df.loc[aoi_list]

        self.labels = None
        if self.path_to["labels"]:
            self.logger.debug(
                "Reading {} file ... ".format(self.path_to["labels"]))
            if self.name.startswith("FL"):
                framelist = loadmat(self.path_to["labels"])
                f1 = framelist[self.name.split(".")[0]][0, 2]
                f2 = framelist[self.name.split(".")[0]][-1, 2]
                self.drift_df = self.drift_df.loc[f1:f2]
                aoi_list = np.unique(framelist[self.name.split(".")[0]][:, 0])
                self.aoi_df = self.aoi_df.loc[aoi_list]
                index = pd.MultiIndex.from_arrays(
                    [framelist[self.name.split(".")[0]][:, 0],
                     framelist[self.name.split(".")[0]][:, 2]],
                    names=["aoi", "frame"])
                self.labels = pd.DataFrame(
                    data=np.zeros((len(self.aoi_df)*len(self.drift_df), 3)),
                    columns=["label", "spotpicker"], index=index)
                self.labels["spotpicker"] = \
                    framelist[self.name.split(".")[0]][:, 1]
                self.labels.loc[self.labels["spotpicker"] == 0,
                                "spotpicker"] = 3
                self.labels.loc[self.labels["spotpicker"] == 2,
                                "spotpicker"] = 0
            else:
                labels_mat = loadmat(self.path_to["labels"])
                index = pd.MultiIndex.from_product(
                    [self.aoi_df.index.values, self.drift_df.index.values],
                    names=["aoi", "frame"])
                self.labels = pd.DataFrame(
                    data=np.zeros((len(self.aoi_df)*len(self.drift_df), 3)),
                    columns=["spotpicker", "probs", "binary"], index=index)
                spot_picker = labels_mat["Intervals"][
                    "CumulativeIntervalArray"][0, 0]
                for sp in spot_picker:
                    aoi = int(sp[-1])
                    start = int(sp[1])
                    end = int(sp[2])
                    if sp[0] in [-2., 0., 2.]:
                        self.labels \
                            .loc[(aoi, start):(aoi, end), "spotpicker"] = 0
                    elif sp[0] in [-3., 1., 3.]:
                        self.labels \
                            .loc[(aoi, start):(aoi, end), "spotpicker"] = 1

            self.labels["label"] = self.labels["spotpicker"]
            self.labels.to_csv(os.path.join(
                self.path, "{}_labels.csv".format(self.xp)))

    def read_glimpse(self, D=14):
        """
        self.target DataFrame
        aoi frame x y abs_x abs_y
        self.drfit DataFrame
        frame dx dy abs_dx abs_dy
        dx = dx % 1
        top_x = int((x - D * 0.5) // 1 + dx // 1)
        x = x - top_x - 1
        """
        self.D, self.height, self.width = \
            D, int(self.header["height"]), int(self.header["width"])
        self.N = len(self.aoi_df)
        self.F = len(self.drift_df)
        # target location
        self.target = pd.DataFrame(
            data={"frame": self.aoi_df["frame"], "x": 0., "y": 0.,
                  "abs_x": self.aoi_df["x"], "abs_y": self.aoi_df["y"]},
            index=self.aoi_df.index)
        # drift
        self.drift = pd.DataFrame(
            data={"dx": 0., "dy": 0., "abs_dx": self.drift_df["dx"],
                  "abs_dy": self.drift_df["dy"]},
            index=self.drift_df.index)
        self._store = np.ones((self.N, self.F, self.D, self.D)) * 2**15
        # loop through each frame
        for i, frame in enumerate(tqdm(self.drift_df.index)):
            # read the entire frame image
            glimpse_number = self.header["filenumber"][frame - 1]
            glimpse_path = os.path.join(
                self.path_to["dir"], "{}.glimpse".format(glimpse_number))
            offset = self.header["offset"][frame - 1]
            with open(glimpse_path, "rb") as fid:
                fid.seek(offset)
                img = np.fromfile(
                    fid, dtype='>i2',
                    count=self.height*self.width) \
                    .reshape(self.height, self.width)

            # new drift list (fractional part)
            self.drift.at[frame, "dx"] = self.drift_df.at[frame, "dx"] % 1
            self.drift.at[frame, "dy"] = self.drift_df.at[frame, "dy"] % 1
            # loop through each aoi
            for j, aoi in enumerate(self.aoi_df.index):
                # top left corner of aoi
                # integer part (target center - half aoi width) \
                # + integer part (drift)
                top_x = int((self.aoi_df.at[aoi, "x"] - self.D * 0.5) // 1
                            + self.drift_df.at[frame, "dx"] // 1)
                left_y = int((self.aoi_df.at[aoi, "y"] - self.D * 0.5) // 1
                             + self.drift_df.at[frame, "dy"] // 1)
                # j-th frame, i-th aoi
                self._store[j, i, :, :] += \
                    img[top_x:top_x+self.D, left_y:left_y+self.D]
                # new target center for the first frame
                #if i == 0:
                #    self.target.at[aoi, "x"] = self.aoi_df.at[aoi, "x"] - top_x - 1
                #    self.target.at[aoi, "y"] = self.aoi_df.at[aoi, "y"] - left_y - 1
        for j, aoi in enumerate(self.aoi_df.index):
            self.target.at[aoi, "x"] = self.aoi_df.at[aoi, "x"] \
                - int((self.aoi_df.at[aoi, "x"] - self.D * 0.5) // 1) - 1
            self.target.at[aoi, "y"] = self.aoi_df.at[aoi, "y"] \
                - int((self.aoi_df.at[aoi, "y"] - self.D * 0.5) // 1) - 1
        # convert data into torch tensor
        self._store = torch.tensor(self._store, dtype=torch.float32)
        torch.save(self._store, os.path.join(
            self.path, "{}.pt".format(self.xp)))
        self.target.to_csv(os.path.join(
            self.path, "{}_target.csv".format(self.xp)))
        self.drift.to_csv(os.path.join(
            self.path, "{}_drift.csv".format(self.xp)))
        # calculate low and high percentiles for imaging
        self.vmin = np.percentile(self._store.cpu(), 5)
        self.vmax = np.percentile(self._store.cpu(), 99)
        data_sorted, _ = self._store.reshape(self.N, self.F, -1).sort(dim=2)
        self.logger.debug("Loaded data from glimpse files")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self._store[idx].to(self.device)

    def __repr__(self):
        return self.path

    def script(self):
        self.read_cfg()
        # convert header into dict format
        mat_header = loadmat(self.path_to["header"])
        self.header = dict()
        for i, dt in enumerate(mat_header["vid"].dtype.names):
            self.header[dt] = np.squeeze(mat_header["vid"][0, 0][i])

        # load driftlist mat file
        drift_mat = loadmat(self.path_to["driftlist"])
        # calculate the cumulative sum of dx and dy
        drift_mat["driftlist"][:, 1:3] = np.cumsum(
            drift_mat["driftlist"][:, 1:3], axis=0)
        # convert driftlist into DataFrame
        self.drift_df = pd.DataFrame(
            drift_mat["driftlist"][:, :3],
            columns=["frame", "dx", "dy"])
        self.drift_df = self.drift_df.astype({"frame": int}).set_index("frame")

        # load aoiinfo mat file
        aoi_mat = loadmat(self.path_to["aoiinfo"])
        # convert aoiinfo into DataFrame
        self.aoi_df = pd.DataFrame(
            aoi_mat["aoiinfo2"],
            columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
        self.aoi_df = self.aoi_df.astype({"aoi": int}).set_index("aoi")
        self.aoi_df["x"] = self.aoi_df["x"] \
            - self.drift_df.at[int(self.aoi_df.at[1, "frame"]), "dx"]
        self.aoi_df["y"] = self.aoi_df["y"] \
            - self.drift_df.at[int(self.aoi_df.at[1, "frame"]), "dy"]


        self.labels = None
        if self.path_to["labels"]:
            if self.name.startswith("FL"):
                framelist = loadmat(self.path_to["labels"])
                f1 = framelist[self.name.split(".")[0]][0, 2]
                f2 = framelist[self.name.split(".")[0]][-1, 2]
                self.drift_df = self.drift_df.loc[f1:f2]
                aoi_list = np.unique(framelist[self.name.split(".")[0]][:, 0])
                self.aoi_df = self.aoi_df.loc[aoi_list]
                index = pd.MultiIndex.from_arrays(
                    [framelist[self.name.split(".")[0]][:, 0],
                     framelist[self.name.split(".")[0]][:, 2]],
                    names=["aoi", "frame"])
                self.labels = pd.DataFrame(
                    data=np.zeros((len(self.aoi_df)*len(self.drift_df), 3)),
                    columns=["spotpicker", "probs", "binary"], index=index)
                self.labels["spotpicker"] = \
                    framelist[self.name.split(".")[0]][:, 1]
                self.labels.loc[self.labels["spotpicker"] == 0,
                                "spotpicker"] = 3
                self.labels.loc[self.labels["spotpicker"] == 2,
                                "spotpicker"] = 0

            self.labels.to_csv(os.path.join(
                self.path, "spotpicker_labels.csv"))
