import math

import os
import pickle

#import qgrid
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from ipywidgets import interact
from matplotlib.patches import Rectangle
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pyro
from pyro.ops.stats import hpdi
from matplotlib.colors import to_rgba_array
import torch.distributions as dist
from torch.distributions.transforms import AffineTransform


def write_summary(epoch_count, epoch_loss, model, svi, writer, feature=False, mcc=False):
    
    writer.add_scalar("ELBO", -epoch_loss, epoch_count)
    if mcc:
        mask = model.data.labels["spotpicker"].values < 2
        k_probs = torch.zeros(model.N,model.F,2)
        k_probs[...,0] = pyro.param("m_probs").squeeze()[...,1] + pyro.param("m_probs").squeeze()[...,3]
        k_probs[...,1] = pyro.param("m_probs").squeeze()[...,2] + pyro.param("m_probs").squeeze()[...,3]
        z_probs = k_probs[...,0] * pyro.param("theta_probs").squeeze()[...,1] + k_probs[...,1] * pyro.param("theta_probs").squeeze()[...,2]
        model.data.labels["probs"] = z_probs.reshape(-1).detach().cpu().numpy()
        model.data.labels["binary"] = model.data.labels["probs"] > 0.5
        writer.add_scalar("MCC", matthews_corrcoef(model.data.labels["spotpicker"].values[mask], model.data.labels["binary"].values[mask]), epoch_count)
    
    for p in pyro.get_param_store().get_all_param_names():
        if pyro.param(p).squeeze().dim() == 0:
            writer.add_scalar(p, pyro.param(p).squeeze().item(), epoch_count)
        elif pyro.param(p).squeeze().dim() == 1:
            if len(pyro.param(p).squeeze()) <= model.K:
                scalars = {str(i): pyro.param(p).squeeze()[i].item() for i in range(pyro.param(p).squeeze().size()[-1])}
                writer.add_scalars("{}".format(p), scalars, epoch_count)
            else:
                writer.add_histogram("{}".format(p), pyro.param(p).squeeze().detach(), epoch_count)
        elif p in ["z_probs", "j_probs", "m_probs"]:
            for i in range(pyro.param(p).squeeze().size()[-1]):
                writer.add_histogram("{}_{}".format(p,i), pyro.param(p).squeeze()[...,i].detach().reshape(-1), epoch_count)
        elif pyro.param(p).squeeze().dim() >= 2:
            for i in range(pyro.param(p).squeeze().size()[0]):
                writer.add_histogram("{}_{}".format(p,i), pyro.param(p).squeeze()[i,...].detach().reshape(-1), epoch_count)

    
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
    
def view_probs(aoi, data, f1, f2, binder, junk, sp):
    plt.figure(figsize=(25,5))
    # height
    if junk: plt.plot(data.drift.index.values[f1:f2+1], data.j_probs[aoi,f1:f2+1,1], marker="o", ms=5, color="C3")
    if binder: plt.plot(data.drift.index.values[f1:f2+1], data.z_probs[aoi,f1:f2+1,1], marker="o", ms=5, color="C2")
    if sp: plt.plot(data.drift.index.values[f1:f2+1], data.l_probs[aoi,f1:f2+1,1], marker="o", ms=5, color="C4")
    #if binder: plt.plot(data.drift.index.values[f1:f2+1], data.m_probs[aoi,f1:f2+1,2], marker="o", ms=5, color="C4")
    plt.ylim(-0.02,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-0.1, data.drift.index.values[f1:f2+1].max()+0.1)
    plt.ylabel("probability", fontsize=25)
    plt.xlabel("frame #", fontsize=25)
    plt.gca().set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.gca().tick_params(axis="x", labelsize=20)
    plt.gca().tick_params(axis="y", labelsize=20)
    #plt.legend()
    plt.tight_layout()
    plt.show()
    
def view_aoi(aoi, frame, data, target, z, m1, m2):
    #fig = plt.figure(figsize=(30,1.5), dpi=600)
    if m1 or m2 or z:
        k_probs = torch.zeros(len(data.drift),2)
        k_probs[:,0] = data.m_probs[aoi,:,0,0,1]+data.m_probs[aoi,:,0,0,3]
        k_probs[:,1] = data.m_probs[aoi,:,0,0,2]+data.m_probs[aoi,:,0,0,3]
    if z:
        z_probs = k_probs[:,0]*data.theta_probs[aoi,:,0,0,1] + k_probs[:,1]*data.theta_probs[aoi,:,0,0,2]

    fig = plt.figure(figsize=(15,3), dpi=600)
    for i in range(20):
        ax = fig.add_subplot(2,10,i+1)
        plt.title("f #{:d}".format(data.drift.index[frame+i]), fontsize=15)
        plt.imshow(data._store[aoi,frame+i].cpu(), cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
        if target:
            plt.plot(data.target.iloc[aoi, 2] + data.drift.iloc[frame+i, 1] + 0.5, data.target.iloc[aoi, 1] + data.drift.iloc[frame+i, 0] + 0.5, "b+", markersize=10, mew=3, alpha=0.7)
        if m1:
            plt.plot(data.target.iloc[aoi, 2] + data.drift.iloc[frame+i, 1] + 0.5 + data.y_mean[aoi,frame+i,0,0,0], data.target.iloc[aoi, 1] + data.drift.iloc[frame+i, 0] + 0.5 + data.x_mean[aoi,frame+i,0,0,0], "C0+", markersize=10, mew=3, alpha=k_probs[frame+i,0])
        if m2:
            plt.plot(data.target.iloc[aoi, 2] + data.drift.iloc[frame+i, 1] + 0.5 + data.y_mean[aoi,frame+i,0,0,1], data.target.iloc[aoi, 1] + data.drift.iloc[frame+i, 0] + 0.5 + data.x_mean[aoi,frame+i,0,0,1], "C1+", markersize=10, mew=3, alpha=k_probs[frame+i,1])
        if z:
            z_color = to_rgba_array("C2", z_probs[frame+i])[0]
            plt.gca().add_patch(Rectangle((0, 0), data.D*z_probs[frame+i], 0.25, edgecolor=z_color, lw=4, facecolor="none"))
            #plt.plot(data.target.iloc[aoi, 2] + data.y_mode[aoi, frame+i] + 0.5, data.target.iloc[aoi, 1] + data.x_mode[aoi, frame+i] + 0.5, "g+", markersize=10, mew=3, alpha=data.z_probs[aoi, frame+i, 1])
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        #plt.gca().axes.xaxis.set_ticklabels([])
        #plt.gca().axes.yaxis.set_ticklabels([])
        #elif show_class == "spotpicker":
        #    rgb_color = np.zeros((3,))
        #    for k in range(2):
        #        rgb_color += to_rgba_array(l_colors[k])[0,:3] * data.l_probs.cpu().numpy()[aoi,frame+i,k]
            #k = data.predictions[aoi,frame+i]
        #    rgb_color = np.where(rgb_color > 1., 1., rgb_color)
        #    plt.gca().add_patch(Rectangle((0, 0), data.D-1, data.D-1, edgecolor=rgb_color, lw=4, facecolor="none"))
    #plt.tight_layout()
    plt.show()
    
def view_aoi_class_summary(data, k):
    plt.figure(figsize=(15,4))

    plt.subplot2grid((1, 4), (0, 0), colspan=2)
    plt.scatter(data.info.index.values, data.info["z_{}".format(k)])
    #plt.scatter(data.info.index, data.info["z"])
    plt.xlabel("frame #")
    plt.ylabel("probability of state {}".format(k))
    
    plt.subplot2grid((1, 4), (0, 2), colspan=1)
    plt.hist(data[torch.tensor((data.info["predictions"] == k).values)].reshape(-1).cpu(), bins=50, alpha=0.3)
    plt.xlabel("pixel intensity")
    plt.ylabel("counts")
    
    plt.subplot2grid((1, 4), (0, 3), colspan=1)
    plt.imshow(data[torch.tensor((data.info["predictions"] == k).values)].mean(dim=0).cpu(),  cmap='gray', vmin=data.vmin, vmax=data.vmax)
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



    

def view_control_summary(aoi, data, f1, f2, m2, m):
    
    b_hpd = hpdi(dist.Gamma(data.c_b_loc[aoi].squeeze() * data.b_beta, data.b_beta).sample((1000,)), 0.95, dim=0)
    #b_hpd = hpdi(dist.Gamma(data.b_loc[aoi].squeeze() * data.D**2, data.D**2).sample((1000,)), 0.95, dim=0)
    b_mean = data.c_b_loc[aoi].squeeze()

    if m2:
        #h_hpd = hpdi(dist.Gamma(data.h_loc[aoi].squeeze(), 1).sample((1000,)), 0.95, dim=0)
        h_hpd = hpdi(dist.Gamma(data.c_h_loc[aoi].squeeze() * data.h_beta, data.h_beta).sample((1000,)), 0.95, dim=0)
        h_mean = data.c_h_loc[aoi].squeeze()

        #scale = torch.sqrt((data.w_mode[aoi].squeeze()**2 + 1/12) / data.h_loc[aoi].squeeze() + 8 * math.pi * data.w_mode[aoi].squeeze()**4 * data.b_loc[aoi,:,0,0].unsqueeze(dim=-1) / data.h_loc[aoi].squeeze()**2)
        #x_hpd = hpdi(dist.Normal(data.x_mode[aoi].squeeze(), scale).sample((1000,)), 0.95, dim=0)
        x_hpd = hpdi(dist.Normal(data.c_x_mean[aoi].squeeze(), data.c_scale[aoi].squeeze()).sample((1000,)), 0.95, dim=0)
        x_mean = data.c_x_mean[aoi].squeeze()

        #y_hpd = hpdi(dist.Normal(data.y_mode[aoi].squeeze(), scale).sample((1000,)), 0.95, dim=0)
        y_hpd = hpdi(dist.Normal(data.c_y_mean[aoi].squeeze(), data.c_scale[aoi].squeeze()).sample((1000,)), 0.95, dim=0)
        y_mean = data.c_y_mean[aoi].squeeze()

        w_hpd = hpdi(Location(data.c_w_mode[aoi].squeeze(), data.c_w_size[aoi].squeeze(), 0.5, 2.5).sample((1000,)), 0.95, dim=0)
        w_mean = data.c_w_mode[aoi].squeeze()

    if m:
        k_probs = np.zeros((len(data.drift),2))
        k_probs[:,0] = data.m_probs[aoi,:,0,0,1]+data.m_probs[aoi,:,0,0,3]
        k_probs[:,1] = data.m_probs[aoi,:,0,0,2]+data.m_probs[aoi,:,0,0,3]
        m_colors = np.zeros((2,len(data.drift),4))
        m_colors[0] += to_rgba_array("C0")
        m_colors[0,:,3] = k_probs[:,0]
        m_colors[1] += to_rgba_array("C1")
        m_colors[1,:,3] = k_probs[:,1]

    plt.figure(figsize=(15,15))
    # height
    plt.subplot(5,1,1)
    for k in range(2):
        if m2 and not m:
            plt.fill_between(data.drift.index.values[f1:f2+1], h_hpd[0][f1:f2+1,k], h_hpd[1][f1:f2+1,k], color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], h_mean[f1:f2+1,k], s=10, color="C{}".format(k))
        if m and m2:
            plt.fill_between(data.drift.index.values[f1:f2+1], h_hpd[0][f1:f2+1,k], h_hpd[1][f1:f2+1,k], where=(k_probs[f1:f2+1,k]>0.5), color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], h_mean[f1:f2+1,k], s=10, color=m_colors[k,f1:f2+1])
    plt.ylim(0,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("intensity", fontsize=20) 

    # x position
    plt.subplot(5,1,2)
    for k in range(2):
        if m2 and not m:
            plt.fill_between(data.drift.index.values[f1:f2+1], x_hpd[0][f1:f2+1,k], x_hpd[1][f1:f2+1,k], color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], x_mean[f1:f2+1,k], s=10, color="C{}".format(k))
        if m and m2:
            plt.fill_between(data.drift.index.values[f1:f2+1], x_hpd[0][f1:f2+1,k], x_hpd[1][f1:f2+1,k], where=(k_probs[f1:f2+1,k]>0.5), color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], x_mean[f1:f2+1,k], s=10, color=m_colors[k,f1:f2+1])
    plt.ylim(-(data.D+3)/2, (data.D+3)/2)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("x position", fontsize=20)
    
    # y position
    plt.subplot(5,1,3)
    for k in range(2):
        if m2 and not m:
            plt.fill_between(data.drift.index.values[f1:f2+1], y_hpd[0][f1:f2+1,k], y_hpd[1][f1:f2+1,k], color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], y_mean[f1:f2+1,k], s=10, color="C{}".format(k))
        if m and m2:
            plt.fill_between(data.drift.index.values[f1:f2+1], y_hpd[0][f1:f2+1,k], y_hpd[1][f1:f2+1,k], where=(k_probs[f1:f2+1,k]>0.5), color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], y_mean[f1:f2+1,k], s=10, color=m_colors[k,f1:f2+1])
    plt.ylim(-(data.D+3)/2, (data.D+3)/2)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("y position", fontsize=20)
    
    # width
    plt.subplot(5,1,4)
    for k in range(2):
        if m2 and not m:
            plt.fill_between(data.drift.index.values[f1:f2+1], w_hpd[0][f1:f2+1,k], w_hpd[1][f1:f2+1,k], color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1,k], s=10, color="C{}".format(k))
        if m and m2:
            plt.fill_between(data.drift.index.values[f1:f2+1], w_hpd[0][f1:f2+1,k], w_hpd[1][f1:f2+1,k], where=(k_probs[f1:f2+1,k]>0.5), color="C{}".format(k), alpha=0.2)
            plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1,k], s=10, color=m_colors[k,f1:f2+1])
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("width", fontsize=20)
    
    # background 
    plt.subplot(5,1,5)
    plt.fill_between(data.drift.index.values[f1:f2+1], b_hpd[0][f1:f2+1], b_hpd[1][f1:f2+1], color="C0", alpha=0.2)
    plt.scatter(data.drift.index.values[f1:f2+1], b_mean[f1:f2+1], s=10, color="C0")
    plt.ylim(0,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("background", fontsize=20)
    plt.xlabel("frame #", fontsize=20)
    
    #plt.legend()
    plt.tight_layout()
    plt.show()
