import math

import os
import pickle

#import qgrid
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pyro
from pyro import param
from pyro.ops.stats import hpdi
from matplotlib.colors import to_rgba_array
import torch.distributions as dist
from torch.distributions.transforms import AffineTransform


def write_summary(epoch_count, epoch_loss, model, svi, writer_scalar, writer_hist, feature=False, mcc=False):
    
    writer_scalar.add_scalar("-ELBO", epoch_loss, epoch_count)
    if mcc:
        mask = model.data.labels["spotpicker"].values < 2
        z_probs = (pyro.param("d/m_probs").squeeze() * pyro.param("d/theta_probs").squeeze()[...,1:].sum(dim=-1)).sum(dim=-1)
        model.data.labels["probs"] = z_probs.reshape(-1).detach().cpu().numpy()
        model.data.labels["binary"] = model.data.labels["probs"] > 0.5
        writer_scalar.add_scalar("MCC", matthews_corrcoef(model.data.labels["spotpicker"].values[mask], model.data.labels["binary"].values[mask]), epoch_count)
    
    for p, value in pyro.get_param_store().named_parameters():
        if pyro.param(p).squeeze().dim() == 0:
            writer_scalar.add_scalar(p, pyro.param(p).squeeze().item(), epoch_count)
        elif pyro.param(p).squeeze().dim() == 1 and pyro.param(p).squeeze().shape[0] <= model.K:
            scalars = {str(i): pyro.param(p).squeeze()[i].item() for i in range(pyro.param(p).squeeze().size()[-1])}
            writer_scalar.add_scalars("{}".format(p), scalars, epoch_count)
        elif pyro.param(p).squeeze().dim() >= 1 and not (epoch_count % 2000):
            writer_hist.add_histogram("{}".format(p), pyro.param(p).squeeze().detach().reshape(-1), epoch_count)

    
def get_offset(frames, header, path_glimpse):
    height = int(header["height"])
    width = int(header["width"])
    offsets = list()
    for frame in frames:
        glimpse_number = header["filenumber"][int(frame-1)]
        with open(os.path.join(path_glimpse, "{}.glimpse".format(glimpse_number))) as fid:
            fid.seek(header['offset'][int(frame-1)])
            img = np.fromfile(fid, dtype='>i2', count=height*width).reshape(height,width)
            img += 2**15
            img.astype(np.uint16)
        offsets.append(np.stack((img[10:70,10:70], img[10:70,-70:-10], img[-70:-10,10:70], img[-70:-10,-70:-10])))
    return np.stack(offsets).reshape(-1)
    
def view_aoi_summary(aoi, data, show_class=False):
    
    if show_class:
        plt.figure(figsize=(15,4))
        rgb_colors = np.zeros((len(data.drift),3))
        for k in data.predictions.unique().tolist():
            rgb_colors += to_rgba_array("C{}".format(k))[:,:3] * data.probs.cpu().numpy()[aoi,:,k].reshape(-1,1)
        rgb_colors = np.where(rgb_colors > 1., 1., rgb_colors)
        plt.scatter(data.drift.index.values, data.intensity[aoi].cpu().numpy(), color=rgb_colors)
        plt.ylim(data.vmin, data.vmax+30)
        plt.xlim(data.drift.index.min()-10, data.drift.index.max()+10)
        plt.xlabel("frame #")
        plt.ylabel("integrated intensity")
        
    else:
        plt.figure(figsize=(15,4))
        plt.subplot2grid((1, 4), (0, 0), colspan=2)
        plt.plot(data.drift.index.values, data.intensity[aoi].cpu().numpy())
        plt.ylim(data.vmin, data.vmax+30)
        plt.xlim(data.drift.index.min()-10, data.drift.index.max()+10)
        plt.xlabel("frame #")
        plt.ylabel("integrated intensity")

        plt.subplot2grid((1, 4), (0, 2), colspan=1)
        plt.hist(data._store[aoi].reshape(-1).cpu(), bins=50)
        plt.xlabel("pixel intensity")
        plt.ylabel("counts")

        plt.subplot2grid((1, 4), (0, 3), colspan=1)
        plt.imshow(data._store[aoi].mean(dim=0).cpu(),  cmap="gray", vmin=data.vmin, vmax=data.vmax)
        plt.title("mean aoi")
    
    plt.tight_layout()
    plt.show()
    

#interact(view_aoi_class_summary, data=fixed(model.data), k=range(model.K))
def view_aoi_class_average(data, K):
    plt.figure(figsize=(3*K,3))

    for k in range(K):
        plt.subplot(1,K,k+1)
        plt.imshow(data._store[data.predictions == k].mean(dim=0).cpu(),  cmap='gray', vmin=data.vmin, vmax=data.vmax)
        plt.title("mean aoi {}".format(k))
    
    plt.tight_layout()
    plt.show()
