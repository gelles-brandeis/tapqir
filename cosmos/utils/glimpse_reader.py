import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Sampler(Dataset):
    def __init__(self,N):
        self.N = N
    
    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        return idx
    

class GlimpseDataset(Dataset):
    """ CoSMoS Dataset """
    
    def __init__(self, D, aoi_df, drift_df, header, path, device, labels=None):
        # store metadata
        self.header, self.path  = header, path
        self.D, self.height, self.width = D, int(self.header["height"]), int(self.header["width"])
        self.N = len(aoi_df)
        self.F = len(drift_df)
        # integrated intensity
        self.intensity = np.zeros((len(aoi_df),len(drift_df)))
        # labels
        self.labels = labels
        try:
            self._store = torch.load(os.path.join(path, "data.pt"), map_location=device)
            assert (self.N, self.F, self.D, self.D) == self._store.shape
            self.target = pd.read_csv(os.path.join(path, "target.csv"), index_col="aoi")
            self.drift = pd.read_csv(os.path.join(path, "drift.csv"), index_col="frame")
            print("aois were read from data.pt, target.csv, and drift.csv files")
        except:
            print("reading aois from glimpse files")
            # target location
            self.target = pd.DataFrame(data={"frame": aoi_df["frame"], "x": 0., "y": 0., "abs_x": aoi_df["x"], "abs_y": aoi_df["y"]}, index=aoi_df.index)
            # drift
            self.drift = pd.DataFrame(data={"dx": 0., "dy": 0., "abs_dx": drift_df["dx"], "abs_dy": drift_df["dy"]}, index=drift_df.index)
            self._store = np.ones((len(aoi_df),len(drift_df), self.D, self.D)) * 2**15
            # loop through each frame
            for i, frame in enumerate(tqdm(drift_df.index)):
                # read the entire frame image
                glimpse_number = self.header["filenumber"][frame - 1]
                glimpse_path = os.path.join(self.path, "{}.glimpse".format(glimpse_number))
                offset = self.header["offset"][frame - 1]
                with open(glimpse_path, "rb") as fid:
                    fid.seek(offset)
                    img = np.fromfile(fid, dtype='>i2', count=self.height*self.width).reshape(self.height, self.width)
                    
                # new drift list (fractional part)
                self.drift.at[frame, "dx"] = drift_df.at[frame, "dx"] % 1 
                self.drift.at[frame, "dy"] = drift_df.at[frame, "dy"] % 1 
                # loop through each aoi
                for j, aoi in enumerate(aoi_df.index):
                    # top left corner of aoi
                    # integer part (target center - half aoi width) + integer part (drift)
                    top_x = int((aoi_df.at[aoi, "x"] - self.D * 0.5) // 1 + drift_df.at[frame, "dx"] // 1)
                    left_y = int((aoi_df.at[aoi, "y"] - self.D * 0.5) // 1 + drift_df.at[frame, "dy"] // 1)
                    # j-th frame, i-th aoi
                    self._store[j,i,:,:] += img[top_x:top_x+self.D, left_y:left_y+self.D]
                    # new target center for the first frame
                    if i == 0:
                        self.target.at[aoi, "x"] = aoi_df.at[aoi, "x"] - top_x - 1
                        self.target.at[aoi, "y"] = aoi_df.at[aoi, "y"] - left_y - 1
            # convert data into torch tensor
            self._store = torch.tensor(self._store, dtype=torch.float32)
            torch.save(self._store, os.path.join(path, "data.pt"))
            self.target.to_csv(os.path.join(path, "target.csv"))
            self.drift.to_csv(os.path.join(path, "drift.csv"))
            print("aois were saved to data.pt, target.csv, and drift.csv files")
        # calculate integrated intensity
        self.intensity = self._store.mean(dim=(2,3))
        # calculate low and high percentiles for imaging
        self.vmin = np.percentile(self._store.cpu(), 5)
        self.vmax = np.percentile(self._store.cpu(), 99)
        data_sorted, _ = self._store.reshape(self.N,self.F,-1).sort(dim=2)
        self.background = data_sorted[...,self.D*2:self.D*4].mean(dim=2)
        self.height = data_sorted[...,-self.D*4:-self.D*2].mean(dim=2) - self.background
        self.background -= self._store.min()
        #self.background -= 90
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        #idx = self.target.index.get_loc(aoi)
        #return AoIDataset(aoi, self._store[idx], self.drift, self.D, self.intensity.loc[:, aoi], self.vmin, self.vmax), aoi
        #return self._store[idx]
        return self._store[idx]
    
    def __repr__(self):
        return self.path

def load_aois(path_glimpse, header, aoi_df, drift_df, aoi_list, frames_list):
    smd = {}
    smd["pixnum"] = int(aoi_df.loc[1,"pixnum"])
    smd["height"] = int(header["vid"]["height"])
    smd["width"] = int(header["vid"]["width"])
    smd["info"] = {}
    smd["aoi_center"] = {}
    smd["aoi_border"] = {}
    smd["data"] = {}
    # get all frames
    all_frames = set([f for sublist in frames_list for f in sublist])
    
    for aoi, frames in zip(aoi_list, frames_list):
        xy_loc = aoi_df.loc[aoi, ["x", "y"]].values #+ drift_df.loc[frames, ["dx", "dy"]].mean().values
        if smd["pixnum"] % 2:
            xy_pixel = np.round(xy_loc)
        else:
            xy_pixel = np.floor(xy_loc) + 0.5
        x1 = int(xy_pixel[0] - (smd["pixnum"] - 1)/2 - 1)
        x2 = int(xy_pixel[0] + (smd["pixnum"] - 1)/2)
        y1 = int(xy_pixel[1] - (smd["pixnum"] - 1)/2 - 1)
        y2 = int(xy_pixel[1] + (smd["pixnum"] - 1)/2)
        smd["aoi_center"][aoi] = xy_loc
        smd["aoi_border"][aoi] = np.array([[x1, x2], [y1, y2]])
        df = pd.DataFrame(data={"aoi": aoi, "frame": frames, "dx": drift_df.loc[frames, "dx"], "dy": drift_df.loc[frames, "dy"], "intensity": 0})
        #print(df.loc[:,["dx", "dy"]].shape, drift_df.loc[frames, ["dx", "dy"]].shape)
        #df["dx"] = drift_df.loc[frames, "dx"]
        df = df.set_index("frame")
        data = np.zeros((len(frames), smd["pixnum"], smd["pixnum"]))
        smd["info"][aoi] = df
        smd["data"][aoi] = data
        
    for frame in all_frames:
        glimpse_number = header["vid"]["filenumber"][0][0][0][int(frame-1)]
        with open(os.path.join(path_glimpse, "{}.glimpse".format(glimpse_number))) as fid:
            fid.seek(header['vid']['offset'][0][0][0][int(frame-1)])
            img = np.fromfile(fid, dtype='>i2', count=smd["height"]*smd["width"]).reshape(smd["height"],smd["width"])
            img += 2**15
            #img = img.astype(np.uint16)
            
            for aoi in smd["info"].keys():
                if frame in smd["info"][aoi].index:
                    x1 = int(smd["aoi_border"][aoi][0,0] + smd["info"][aoi].loc[frame, "dx"] // 1)
                    x2 = int(smd["aoi_border"][aoi][0,1] + smd["info"][aoi].loc[frame, "dx"] // 1)
                    y1 = int(smd["aoi_border"][aoi][1,0] + smd["info"][aoi].loc[frame, "dy"] // 1)
                    y2 = int(smd["aoi_border"][aoi][1,1] + smd["info"][aoi].loc[frame, "dy"] // 1)
                    smd["data"][aoi][smd["info"][aoi].index.get_loc(frame),:,:] = img[x1:x2,y1:y2]

                    smd["info"][aoi].at[frame, "intensity"] = img[x1:x2,y1:y2].sum()
                    smd["info"][aoi].loc[frame, "dx"] = smd["info"][aoi].loc[frame, "dx"] % 1
                    smd["info"][aoi].loc[frame, "dy"] = smd["info"][aoi].loc[frame, "dy"] % 1
               
    for aoi in smd["data"].keys():
        smd["data"][aoi] = torch.tensor(smd["data"][aoi], dtype=torch.float32)
            
    return smd
