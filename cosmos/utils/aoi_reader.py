import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tqdm import tqdm
import configparser
import logging

from cosmos.utils.glimpse_reader import GlimpseDataset

def ReadAoi(dataset, device):
    logging.info("*** {} ***".format(dataset))
    logging.info("reading config.ini for {} ... ".format(dataset))
    config = configparser.ConfigParser(allow_no_value=True)
    config.read("../config.ini")
    assert dataset in config

    path_header = config[dataset]["path_header"]
    path = config[dataset]["path"]
    aoi_filename = config[dataset]["aoi_filename"]
    drift_filename = config[dataset]["drift_filename"]
    labels_filename = config[dataset]["labels_filename"]
    logging.info("done")

    # convert header into dict format
    logging.info("reading header.mat file ... ")
    mat_header = loadmat(os.path.join(path_header, "header.mat"))
    header = dict()
    for i, dt in  enumerate(mat_header["vid"].dtype.names):
        header[dt] = np.squeeze(mat_header["vid"][0,0][i])
    logging.info("done")


    # load driftlist mat file
    logging.info("reading {} file ... ".format(drift_filename))
    drift_mat = loadmat(os.path.join(path, drift_filename))
    # calculate the cumulative sum of dx and dy
    logging.info("calculating cumulative drift ... ")
    drift_mat["driftlist"][:, 1:3] = np.cumsum(
        drift_mat["driftlist"][:, 1:3], axis=0)
    # convert driftlist into DataFrame
    drift_df = pd.DataFrame(drift_mat["driftlist"][:,:3], columns=["frame", "dx", "dy"])
    #drift_df = pd.DataFrame(drift_mat["driftlist"], columns=["frame", "dx", "dy", "timestamp"])
    drift_df = drift_df.astype({"frame": int}).set_index("frame")
    logging.info("done")

    # load aoiinfo mat file
    logging.info("reading {} file ... ".format(aoi_filename))
    aoi_mat = loadmat(os.path.join(path, aoi_filename))
    # convert aoiinfo into DataFrame
    if dataset in ["Gracecy3"]:
        aoi_df = pd.DataFrame(aoi_mat["aoifits"]["aoiinfo2"][0,0], columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
    else:
        aoi_df = pd.DataFrame(aoi_mat["aoiinfo2"], columns=["frame", "ave", "x", "y", "pixnum", "aoi"])
    aoi_df = aoi_df.astype({"aoi": int}).set_index("aoi")
    logging.info("adjusting target position from frame {} to frame 1 ... ".format(aoi_df.at[1, "frame"]))
    aoi_df["x"] = aoi_df["x"] - drift_df.at[int(aoi_df.at[1, "frame"]), "dx"]
    aoi_df["y"] = aoi_df["y"] - drift_df.at[int(aoi_df.at[1, "frame"]), "dy"]
    logging.info("done")


    if dataset in ["FL_1_1117_0OD", "FL_3339_4444_0p8OD"]:
        framelist = loadmat("/home/ordabayev/Documents/Datasets/Bayesian_test_files/B33p44a_FrameList_files.dat")
        f1 = framelist[dataset][0,2]
        f2 = framelist[dataset][-1,2]
        drift_df = drift_df.loc[f1:f2]
        aoi_list = np.unique(framelist[dataset][:,0])
        aoi_df = aoi_df.loc[aoi_list]
        labels = pd.DataFrame(data=framelist[dataset], columns=["aoi", "detected", "frame"])
    elif dataset in ["LarryCy3sigma54Short", "LarryCy3sigma54NegativeControlShort"]:
        f1 = 170
        f2 = 1000 #4576
        drift_df = drift_df.loc[f1:f2]
        #aoi_list = np.array([2,4,8,10,11,14,15,18,19,20,21,23,24,25,26,32])
        aoi_list = np.arange(1,33)
        aoi_df = aoi_df.loc[aoi_list]
        print("reading labels ...", end="")
        #labels_mat = loadmat("/home/ordabayev/Documents/Datasets/Larry-Cy3-sigma54/b27p131g_specific_Intervals.dat")
    elif dataset in ["Gracecy3Short"]:
        aoi_list = np.arange(160,240)
        aoi_df = aoi_df.loc[aoi_list]

    labels = None
    if labels_filename:
        print("reading {} file ... ".format(labels_filename), end="")
        labels_mat = loadmat(os.path.join(path, labels_filename))
        index = pd.MultiIndex.from_product([aoi_df.index.values, drift_df.index.values], names=["aoi", "frame"])
        labels = pd.DataFrame(data=np.zeros((len(aoi_df)*len(drift_df),3)), columns=["spotpicker", "probs", "binary"], index=index)
        spot_picker = labels_mat["Intervals"]["CumulativeIntervalArray"][0,0]
        for sp in spot_picker:
            aoi = int(sp[-1])
            start = int(sp[1])
            end = int(sp[2])
            if sp[0] in [-2., 0., 2.]:
                labels.loc[(aoi,start):(aoi,end), "spotpicker"] = 0
            elif sp[0] in [-3., 1., 3.]:
                labels.loc[(aoi,start):(aoi,end), "spotpicker"] = 1
        print("done")

    print("\nsaving drift_df.csv, {}_aoi_df.csv, {}_labels.csv files ..., ".format(dataset,dataset), end="")
    drift_df.to_csv(os.path.join(path_header, "drift_df.csv"))
    aoi_df.to_csv(os.path.join(path_header, "{}_aoi_df.csv".format(dataset)))
    if labels_filename: labels.to_csv(os.path.join(path_header, "{}_labels.csv".format(dataset)))
    print("done")

    data = GlimpseDataset(dataset, D=14, aoi_df=aoi_df, drift_df=drift_df, header=header, path=path_header, device=device, labels=labels)

    return data
