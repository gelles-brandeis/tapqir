import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tapqir.utils.dataset import CosmosDataset, GlimpseDataset


def read_glimpse(path, D):

    """
    target DataFrame
    aoi frame x y abs_x abs_y
    drfit DataFrame
    frame dx dy abs_dx abs_dy
    dx = dx % 1
    top_x = int((x - D * 0.5) // 1 + dx // 1)
    x = x - top_x - 1
    """
    device = torch.device("cpu")
    glimpse = GlimpseDataset(path)
    # drift
    drift = pd.DataFrame(
        data={
            "dx": 0.0,
            "dy": 0.0,
            "abs_dx": glimpse.drift_df["dx"],
            "abs_dy": glimpse.drift_df["dy"],
        },
        index=glimpse.drift_df.index,
    )

    target = {}
    data = {}
    for dtype in glimpse.dtypes:
        # target location
        target[dtype] = pd.DataFrame(
            data={
                "frame": glimpse.aoi_df[dtype]["frame"],
                "x": 0.0,
                "y": 0.0,
                "abs_x": glimpse.aoi_df[dtype]["x"],
                "abs_y": glimpse.aoi_df[dtype]["y"],
            },
            index=glimpse.aoi_df[dtype].index,
        )
        data[dtype] = np.zeros(
            (len(glimpse.aoi_df[dtype]), len(glimpse.drift_df), D, D)
        )
    offsets = np.zeros((len(drift), 4, 30, 30))
    # loop through each frame
    for i, frame in enumerate(tqdm(drift.index)):
        img = glimpse[frame]

        offsets[i, 0, :, :] += img[10:40, 10:40]
        offsets[i, 1, :, :] += img[10:40, -40:-10]
        offsets[i, 2, :, :] += img[-40:-10, 10:40]
        offsets[i, 3, :, :] += img[-40:-10, -40:-10]
        # new drift list (fractional part)
        drift.at[frame, "dx"] = glimpse.drift_df.at[frame, "dx"] % 1
        drift.at[frame, "dy"] = glimpse.drift_df.at[frame, "dy"] % 1
        for dtype in glimpse.dtypes:
            # loop through each aoi
            for j, aoi in enumerate(glimpse.aoi_df[dtype].index):
                # top left corner of aoi
                # integer part (target center - half aoi width) \
                # + integer part (drift)
                top_x = int(
                    (glimpse.aoi_df[dtype].at[aoi, "x"] - D * 0.5) // 1
                    + glimpse.drift_df.at[frame, "dx"] // 1
                )
                left_y = int(
                    (glimpse.aoi_df[dtype].at[aoi, "y"] - D * 0.5) // 1
                    + glimpse.drift_df.at[frame, "dy"] // 1
                )
                # j-th frame, i-th aoi
                data[dtype][j, i, :, :] += img[top_x : top_x + D, left_y : left_y + D]
                # new target center for the first frame
                # if i == 0:
                #    target.at[aoi, "x"] = aoi_df.at[aoi, "x"] - top_x - 1
                #    target.at[aoi, "y"] = aoi_df.at[aoi, "y"] - left_y - 1
    offset = torch.tensor(offsets, dtype=torch.float32)
    for dtype in glimpse.dtypes:
        for j, aoi in enumerate(glimpse.aoi_df[dtype].index):
            target[dtype].at[aoi, "x"] = (
                glimpse.aoi_df[dtype].at[aoi, "x"]
                - int((glimpse.aoi_df[dtype].at[aoi, "x"] - D * 0.5) // 1)
                - 1
            )
            target[dtype].at[aoi, "y"] = (
                glimpse.aoi_df[dtype].at[aoi, "y"]
                - int((glimpse.aoi_df[dtype].at[aoi, "y"] - D * 0.5) // 1)
                - 1
            )
        # convert data into torch tensor
        data[dtype] = torch.tensor(data[dtype], dtype=torch.float32)
        dataset = CosmosDataset(
            data[dtype],
            target[dtype],
            drift,
            dtype,
            device,
            glimpse.labels[dtype],
            offset,
        )
        dataset.save(path)
