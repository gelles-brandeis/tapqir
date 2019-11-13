import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tqdm import tqdm
import configparser

from cosmos.utils.glimpse_reader import GlimpseDataset

def ReadAoi(dataset, device):
    print("reading config.ini for {} ... ".format(dataset), end="")
    config = configparser.ConfigParser()
    config.read("config.ini")
    assert dataset in config

    path_header = config[dataset]["path_header"]
    path = config[dataset]["path"]
    aoi_filename = config[dataset]["aoi_filename"]
    drift_filename = config[dataset]["drift_filename"]
    print("done")

    # convert header into dict format
    print("reading header ... ", end="")
    mat_header = loadmat(os.path.join(path_header, "header.mat"))
    header = dict()
    for i, dt in  enumerate(mat_header["vid"].dtype.names):
        header[dt] = np.squeeze(mat_header["vid"][0,0][i])
    print("done")


    # load driftlist mat file
    print("reading drift file ... ", end="")
    drift_mat = loadmat(os.path.join(path, drift_filename))
    # calculate the cumulative sum of dx and dy
    print("calculating cumulative drift ... ", end="")
    drift_mat["driftlist"][:, 1:3] = np.cumsum(
        drift_mat["driftlist"][:, 1:3], axis=0)
    # convert driftlist into DataFrame
    drift_df = pd.DataFrame(drift_mat["driftlist"][:,:3], columns=["frame", "dx", "dy"])
    #drift_df = pd.DataFrame(drift_mat["driftlist"], columns=["frame", "dx", "dy", "timestamp"])
    drift_df = drift_df.astype({"frame": int}).set_index("frame")
    print("done")

    # load aoiinfo mat file
    print("reading aoiinfo file ... ", end="")
    aoi_mat = loadmat(os.path.join(path, aoi_filename))
    # convert aoiinfo into DataFrame
    if dataset in ["Gracecy3"]:
        aoi_df = pd.DataFrame(aoi_mat["aoifits"]["aoiinfo2"][0,0], columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
    else:
        aoi_df = pd.DataFrame(aoi_mat["aoiinfo2"], columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
    aoi_df = aoi_df.astype({"aoi": int}).set_index("aoi")
    print("adjusting target position from frame {} to frame 1 ... ".format(aoi_df.at[1, "frame"]), end="")
    aoi_df["x"] = aoi_df["x"] - drift_df.at[int(aoi_df.at[1, "frame"]), "dx"]
    aoi_df["y"] = aoi_df["y"] - drift_df.at[int(aoi_df.at[1, "frame"]), "dy"]
    print("done")


    labels = None
    if dataset in ["FL_1_1117_0OD", "FL_3339_4444_0p8OD"]:
        framelist = loadmat("/home/ordabayev/Documents/Datasets/Bayesian_test_files/B33p44a_FrameList_files.dat")
        f1 = framelist[dataset][0,2]
        f2 = framelist[dataset][-1,2]
        drift_df = drift_df.loc[f1:f2]
        aoi_list = np.unique(framelist[dataset][:,0])
        aoi_df = aoi_df.loc[aoi_list]
        labels = pd.DataFrame(data=framelist[dataset], columns=["aoi", "detected", "frame"])
    elif dataset in ["LarryCy3sigma54"]:
        f1 = 170
        f2 = 4576 #1000 #
        drift_df = drift_df.loc[f1:f2]
        #aoi_list = np.array([2,4,8,10,11,14,15,18,19,20,21,23,24,25,26,32])
        #aoi_df = aoi_df.loc[aoi_list]

    print("saving drift_df.csv and aoi_df.csv files ..., ", end="")
    drift_df.to_csv(os.path.join(path_header, "drift_df.csv"))
    aoi_df.to_csv(os.path.join(path_header, "aoi_df.csv"))
    print("done")
    #drift_df.head(6)

    data = GlimpseDataset(D=14, aoi_df=aoi_df, drift_df=drift_df, header=header, path=path_header, device=device, labels=labels)

    return data
