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
from matplotlib.colors import to_rgba_array
import torch.distributions as dist
from torch.distributions.transforms import AffineTransform


def write_summary(epoch_count, epoch_loss, model, svi, writer, feature=False, mcc=False):
    
    #if feature:
    #    epoch_loss = svi.evaluate_loss(model.data[:])
    #else:
    #    epoch_loss = svi.evaluate_loss(
    #            torch.arange(model.N), torch.arange(model.F))
    writer.add_scalar("ELBO", -epoch_loss, epoch_count)
    if model.data.labels:
        mask = model.data.labels["detected"].values < 2
        predictions = pyro.param("z_probs").squeeze().detach().cpu().argmax(dim=2).reshape(-1)[mask]
        labels = model.data.labels["detected"].values[mask]
        writer.add_scalar("MCC", matthews_corrcoef(labels, predictions), epoch_count)
    #if not feature:
    #    writer.add_scalar("Uncertain", len(torch.nonzero((pyro.param("z_probs").cpu().squeeze()[...,0] > 0.3) & (pyro.param("z_probs").cpu().squeeze()[...,0] < 0.7))[:,0].unique()), epoch_count)
    
    for p in pyro.get_param_store().get_all_param_names():
        if pyro.param(p).squeeze().dim() == 0:
            writer.add_scalar(p, pyro.param(p).squeeze().item(), epoch_count)
        elif pyro.param(p).squeeze().dim() == 1:
            if len(pyro.param(p).squeeze()) <= model.K:
                scalars = {str(i): pyro.param(p).squeeze()[i].item() for i in range(pyro.param(p).squeeze().size()[-1])}
                writer.add_scalars("{}".format(p), scalars, epoch_count)
            else:
                writer.add_histogram("{}".format(p), pyro.param(p).squeeze().detach(), epoch_count)
        elif p in ["z_probs", "j_probs"]:
            for i in range(pyro.param(p).squeeze().size()[-1]):
                writer.add_histogram("{}_{}".format(p,i), pyro.param(p).squeeze()[...,i].detach().reshape(-1), epoch_count)
        elif pyro.param(p).squeeze().dim() >= 2:
            for i in range(pyro.param(p).squeeze().size()[0]):
                writer.add_histogram("{}_{}".format(p,i), pyro.param(p).squeeze()[i,...].detach().reshape(-1), epoch_count)

def save_obj(obj, name):
    with ope(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def view_glimpse(frame, aoi, aoi_df, drift_df, header, path_glimpse, selected_aoi, all_aois, label, offset):
    height = int(header["height"])
    width = int(header["width"])
    glimpse_number = header["filenumber"][int(frame-1)]
    with open(os.path.join(path_glimpse, "{}.glimpse".format(glimpse_number))) as fid:
        fid.seek(header['offset'][int(frame-1)])
        img = np.fromfile(fid, dtype='>i2', count=height*width).reshape(height,width)
        img += 2**15
        img.astype(np.uint16)
    plt.figure(figsize=(15,7.5))
    plt.imshow(img,  cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    if all_aois:
        for j in aoi_df.index.values:
            y_pos = aoi_df.at[j, "abs_y"] + drift_df.at[frame, "abs_dy"] - 5
            x_pos = aoi_df.at[j, "abs_x"] + drift_df.at[frame, "abs_dx"] - 5
            plt.gca().add_patch(Rectangle((y_pos, x_pos), 10, 10, edgecolor="b", facecolor="none"))
            if label:
                plt.gca().text(y_pos, x_pos, str(j), fontsize=10, color="white")
    if selected_aoi:
        y_pos = aoi_df.at[aoi, "abs_y"] + drift_df.at[frame, "abs_dy"] - 5
        x_pos = aoi_df.at[aoi, "abs_x"] + drift_df.at[frame, "abs_dx"] - 5
        plt.gca().add_patch(Rectangle((y_pos, x_pos), 10, 10, edgecolor="r", facecolor="none"))
        
    if offset:
        plt.gca().add_patch(Rectangle((0, 0), 70, 70, edgecolor="g", facecolor="none"))
        plt.gca().add_patch(Rectangle((0, height-70), 70, 70, edgecolor="g", facecolor="none"))
        plt.gca().add_patch(Rectangle((width-70, 0), 70, 70, edgecolor="g", facecolor="none"))
        plt.gca().add_patch(Rectangle((width-70, height-70), 70, 70, edgecolor="g", facecolor="none"))
        #np.stack((img[:70,:70], img[:70,-70:], img[-70:,:70], img[-70:,-70:])).reshape(-1)
        
    plt.show()
    
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

def view_theta(aoi, data, f1, f2, theta1, theta2, m):
    if m:
        data.m_probs = np.zeros((data.N,data.F,3))
        data.m_probs[:,:,0] = data.z_probs[:,:,0] * data.j_probs[:,:,0]
        data.m_probs[:,:,2] = data.z_probs[:,:,1] * data.j_probs[:,:,1]
        data.m_probs[:,:,1] = 1 - data.m_probs[:,:,0] - data.m_probs[:,:,2]
        m_colors = np.zeros((len(data.drift),4)) + to_rgba_array("C0")
        m_colors[:,3] = data.m_probs[aoi,:,1]
        m_colors_1 = np.zeros((len(data.drift),4)) + to_rgba_array("C3")
        m_colors_1[:,3] = data.m_probs[aoi,:,2]
        m_colors_2 = np.zeros((len(data.drift),4)) + to_rgba_array("C2")
        m_colors_2[:,3] = data.m_probs[aoi,:,2]

    plt.figure(figsize=(25,5))

    # height
    if theta1: plt.scatter(data.drift.index.values[f1:f2+1], data.theta_probs[aoi,f1:f2+1,0,0,1,0], s=10, color="C3")
    if theta2: plt.scatter(data.drift.index.values[f1:f2+1], data.theta_probs[aoi,f1:f2+1,0,0,1,1], s=10, color="C2")
    if m: plt.scatter(data.drift.index.values[f1:f2+1], data.theta_probs[aoi,f1:f2+1,0,0,1,0], s=10, color=m_colors_1[f1:f2+1])
    if m: plt.scatter(data.drift.index.values[f1:f2+1], data.theta_probs[aoi,f1:f2+1,0,0,1,1], s=10, color=m_colors_2[f1:f2+1])
    #if m: plt.plot(data.drift.index.values[f1:f2+1], data.m_probs[aoi,f1:f2+1,1], marker="o", ms=5, color="C3")
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

def view_junk_summary(aoi, data, f1, f2, m1, m2, m):
    
    if m:
        data.m_probs = np.zeros((data.N,data.F,3))
        data.m_probs[:,:,0] = data.z_probs[:,:,0] * data.j_probs[:,:,0]
        data.m_probs[:,:,2] = data.z_probs[:,:,1] * data.j_probs[:,:,1]
        data.m_probs[:,:,1] = 1 - data.m_probs[:,:,0] - data.m_probs[:,:,2]
    #b_hpd1, b_hpd2 = hpd(dist.Gamma(data.b_loc[aoi] * data.b_beta[aoi], data.b_beta[aoi]).sample((1000,)))
    b_hpd1, b_hpd2 = hpd(dist.Gamma(data.b_loc[aoi,:,0,0] * data.b_beta, data.b_beta).sample((1000,)))
    b_mean = data.b_loc[aoi,:,0,0].cpu().numpy()

    if m1:
        h_hpd1, h_hpd2 = hpd(dist.Gamma(data.h_loc[aoi,:,0,0,0,0] * data.h_beta[aoi,:,0,0,0,0], data.h_beta[aoi,:,0,0,0,0]).sample((1000,)))
        #h_hpd1, h_hpd2 = hpd(dist.Gamma(data.h_loc * data.size[aoi,:,0,0,0,0] * data.h_beta[aoi,:,0,0,0,0], data.h_beta[aoi,:,0,0,0,0]).sample((1000,)))
        h_mean = data.h_loc[aoi,:,0,0,0,0].cpu().numpy()

        x_hpd1, x_hpd2 = hpd(Location(data.x_mode[aoi,:,0,0,0,0], data.size[aoi,:,0,0,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #x_hpd1, x_hpd2 = hpd(Location(data.x_mode[aoi,:,0,0,0,0], data.x_size[aoi,:,0,0,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        x_mean = data.x_mode[aoi,:,0,0,0,0].cpu().numpy()

        y_hpd1, y_hpd2 = hpd(Location(data.y_mode[aoi,:,0,0,0,0], data.size[aoi,:,0,0,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #y_hpd1, y_hpd2 = hpd(Location(data.y_mode[aoi,:,0,0,0,0], data.y_size[aoi,:,0,0,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        y_mean = data.y_mode[aoi,:,0,0,0,0].cpu().numpy()

        #w_hpd1, w_hpd2 = hpd(Location(data.w_mode, data.w_size[aoi,:,0,0,0,0], 0.5, 3.).sample((1000,)))
        #w_mean = data.w_loc.cpu().numpy()
        #w_hpd1, w_hpd2 = hpd(dist.Gamma(data.w_loc[aoi,:,0,0,0,0] * data.w_beta[aoi,:,0,0,0,0], data.w_beta[aoi,:,0,0,0,0]).sample((1000,)))
        #w_mean = data.w_mode.cpu().numpy()

    if m2:
        jh_hpd1_1, jh_hpd2_1 = hpd(dist.Gamma(data.h_loc[aoi,:,0,0,1,0] * data.h_beta[aoi,:,0,0,1,0], data.h_beta[aoi,:,0,0,1,0]).sample((1000,)))
        #jh_hpd1_1, jh_hpd2_1 = hpd(dist.Gamma(data.h_loc * data.size[aoi,:,0,0,1,0] * data.h_beta[aoi,:,0,0,1,0], data.h_beta[aoi,:,0,0,1,0]).sample((1000,)))
        jh_mean_1 = data.h_loc[aoi,:,0,0,1,0].cpu().numpy()

        jx_hpd1_1, jx_hpd2_1 = hpd(Location(data.x_mode[aoi,:,0,0,1,0], data.size[aoi,:,0,0,1,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #jx_hpd1_1, jx_hpd2_1 = hpd(Location(data.x_mode[aoi,:,0,0,1,0], data.x_size[aoi,:,0,0,1,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jx_mean_1 = data.x_mode[aoi,:,0,0,1,0].cpu().numpy()

        jy_hpd1_1, jy_hpd2_1 = hpd(Location(data.y_mode[aoi,:,0,0,1,0], data.size[aoi,:,0,0,1,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #jy_hpd1_1, jy_hpd2_1 = hpd(Location(data.y_mode[aoi,:,0,0,1,0], data.y_size[aoi,:,0,0,1,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jy_mean_1 = data.y_mode[aoi,:,0,0,1,0].cpu().numpy()

        #jw_hpd1_1, jw_hpd2_1 = hpd(Location(data.w_mode, data.w_size[aoi,:,0,0,1,0], 0.5, 3.).sample((1000,)))
        #jw_mean_1 = data.w_loc.cpu().numpy()
        #jw_hpd1_1, jw_hpd2_1 = hpd(dist.Gamma(data.w_loc[aoi,:,0,0,1,0] * data.w_beta[aoi,:,0,0,1,0], data.w_beta[aoi,:,0,0,1,0]).sample((1000,)))
        #jw_mean_1 = data.w_mode.cpu().numpy()

        jh_hpd1_2, jh_hpd2_2 = hpd(dist.Gamma(data.h_loc[aoi,:,0,0,1,1] * data.h_beta[aoi,:,0,0,1,1], data.h_beta[aoi,:,0,0,1,1]).sample((1000,)))
        #jh_hpd1_2, jh_hpd2_2 = hpd(dist.Gamma(data.h_loc * data.size[aoi,:,0,0,1,1] * data.h_beta[aoi,:,0,0,1,1], data.h_beta[aoi,:,0,0,1,1]).sample((1000,)))
        jh_mean_2 = data.h_loc[aoi,:,0,0,1,1].cpu().numpy()

        jx_hpd1_2, jx_hpd2_2 = hpd(Location(data.x_mode[aoi,:,0,0,1,1], data.size[aoi,:,0,0,1,1], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #jx_hpd1_2, jx_hpd2_2 = hpd(Location(data.x_mode[aoi,:,0,0,1,1], data.x_size[aoi,:,0,0,1,1], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jx_mean_2 = data.x_mode[aoi,:,0,0,1,1].cpu().numpy()

        jy_hpd1_2, jy_hpd2_2 = hpd(Location(data.y_mode[aoi,:,0,0,1,1], data.size[aoi,:,0,0,1,1], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #jy_hpd1_2, jy_hpd2_2 = hpd(Location(data.y_mode[aoi,:,0,0,1,1], data.y_size[aoi,:,0,0,1,1], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jy_mean_2 = data.y_mode[aoi,:,0,0,1,1].cpu().numpy()

        #jw_hpd1_2, jw_hpd2_2 = hpd(Location(data.w_mode, data.w_size[aoi,:,0,0,1,1], 0.5, 3.).sample((1000,)))
        #jw_mean_2 = data.w_loc.cpu().numpy()
        #jw_hpd1_2, jw_hpd2_2 = hpd(dist.Gamma(data.w_loc[aoi,:,0,0,1,1] * data.w_beta[aoi,:,0,0,1,1], data.w_beta[aoi,:,0,0,1,1]).sample((1000,)))
        #jw_mean_2 = data.w_mode.cpu().numpy()

    if m:
        m_colors = np.zeros((len(data.drift),4)) + to_rgba_array("C0")
        m_colors[:,3] = data.m_probs[aoi,:,1]
        m_colors_1 = np.zeros((len(data.drift),4)) + to_rgba_array("C3")
        m_colors_1[:,3] = data.m_probs[aoi,:,2]
        m_colors_2 = np.zeros((len(data.drift),4)) + to_rgba_array("C2")
        m_colors_2[:,3] = data.m_probs[aoi,:,2]

    plt.figure(figsize=(15,15))
    # height
    plt.subplot(5,1,1)
    if m1 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], h_hpd1[f1:f2+1], h_hpd2[f1:f2+1], color="C0", alpha=0.2)
    if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jh_hpd1_1[f1:f2+1], jh_hpd2_1[f1:f2+1], color="C3", alpha=0.2)
    if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jh_hpd1_2[f1:f2+1], jh_hpd2_2[f1:f2+1], color="C2", alpha=0.2)
    if m and m1: plt.fill_between(data.drift.index.values[f1:f2+1], h_hpd1[f1:f2+1], h_hpd2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,1]>0.5), color="C0", alpha=0.2)
    if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jh_hpd1_1[f1:f2+1], jh_hpd2_1[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C3", alpha=0.2)
    if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jh_hpd1_2[f1:f2+1], jh_hpd2_2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C2", alpha=0.2)

    if m1 and not m: plt.scatter(data.drift.index.values[f1:f2+1], h_mean[f1:f2+1], s=10, color="C0")
    if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jh_mean_1[f1:f2+1], s=10, color="C3")
    if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jh_mean_2[f1:f2+1], s=10, color="C2")
    if m and m1: plt.scatter(data.drift.index.values[f1:f2+1], h_mean[f1:f2+1], s=10, color=m_colors[f1:f2+1])
    if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jh_mean_1[f1:f2+1], s=10, color=m_colors_1[f1:f2+1])
    if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jh_mean_2[f1:f2+1], s=10, color=m_colors_2[f1:f2+1])
    plt.ylim(0,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("intensity", fontsize=20) 

    # x position
    plt.subplot(5,1,2)
    if m1 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], x_hpd1[f1:f2+1], x_hpd2[f1:f2+1], color="C0", alpha=0.2)
    if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jx_hpd1_1[f1:f2+1], jx_hpd2_1[f1:f2+1], color="C3", alpha=0.2)
    if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jx_hpd1_2[f1:f2+1], jx_hpd2_2[f1:f2+1], color="C2", alpha=0.2)
    if m and m1: plt.fill_between(data.drift.index.values[f1:f2+1], x_hpd1[f1:f2+1], x_hpd2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,1]>0.5), color="C0", alpha=0.2)
    if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jx_hpd1_1[f1:f2+1], jx_hpd2_1[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C3", alpha=0.2)
    if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jx_hpd1_2[f1:f2+1], jx_hpd2_2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C2", alpha=0.2)

    if m1 and not m: plt.scatter(data.drift.index.values[f1:f2+1], x_mean[f1:f2+1], s=10, color="C0")
    if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jx_mean_1[f1:f2+1], s=10, color="C3")
    if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jx_mean_2[f1:f2+1], s=10, color="C2")
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fx_mean[f1:f2+1], s=10, color="C0")
    if m and m1: plt.scatter(data.drift.index.values[f1:f2+1], x_mean[f1:f2+1], s=10, color=m_colors[f1:f2+1])
    if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jx_mean_1[f1:f2+1], s=10, color=m_colors_1[f1:f2+1])
    if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jx_mean_2[f1:f2+1], s=10, color=m_colors_2[f1:f2+1])
    plt.ylim(-(data.D+3)/2, (data.D+3)/2)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("x position", fontsize=20)
    
    # y position
    plt.subplot(5,1,3)
    if m1 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], y_hpd1[f1:f2+1], y_hpd2[f1:f2+1], color="C0", alpha=0.2)
    if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jy_hpd1_1[f1:f2+1], jy_hpd2_1[f1:f2+1], color="C3", alpha=0.2)
    if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jy_hpd1_2[f1:f2+1], jy_hpd2_2[f1:f2+1], color="C2", alpha=0.2)
    if m and m1: plt.fill_between(data.drift.index.values[f1:f2+1], y_hpd1[f1:f2+1], y_hpd2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,1]>0.5), color="C0", alpha=0.2)
    if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jy_hpd1_1[f1:f2+1], jy_hpd2_1[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C3", alpha=0.2)
    if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jy_hpd1_2[f1:f2+1], jy_hpd2_2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C2", alpha=0.2)

    if m1 and not m: plt.scatter(data.drift.index.values[f1:f2+1], y_mean[f1:f2+1], s=10, color="C0")
    if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jy_mean_1[f1:f2+1], s=10, color="C3")
    if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jy_mean_2[f1:f2+1], s=10, color="C2")
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fy_mean[f1:f2+1], s=10, color="C0")
    if m and m1: plt.scatter(data.drift.index.values[f1:f2+1], y_mean[f1:f2+1], s=10, color=m_colors[f1:f2+1])
    if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jy_mean_1[f1:f2+1], s=10, color=m_colors_1[f1:f2+1])
    if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jy_mean_2[f1:f2+1], s=10, color=m_colors_2[f1:f2+1])
    plt.ylim(-(data.D+3)/2, (data.D+3)/2)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("y position", fontsize=20)
    
    # width
    plt.subplot(5,1,4)
    #if m1 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], w_hpd1[f1:f2+1], w_hpd2[f1:f2+1], color="C0", alpha=0.2)
    #if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jw_hpd1_1[f1:f2+1], jw_hpd2_1[f1:f2+1], color="C2", alpha=0.2)
    #if m2 and not m: plt.fill_between(data.drift.index.values[f1:f2+1], jw_hpd1_2[f1:f2+1], jw_hpd2_2[f1:f2+1], color="C3", alpha=0.2)
    #if feature: plt.fill_between(data.drift.index.values[f1:f2+1], fw_hpd1[f1:f2+1], fw_hpd2[f1:f2+1], color="C0", alpha=0.2)
    #if m and m1: plt.fill_between(data.drift.index.values[f1:f2+1], w_hpd1[f1:f2+1], w_hpd2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,1]>0.5), color="C0", alpha=0.2)
    #if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jw_hpd1_1[f1:f2+1], jw_hpd2_1[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C2", alpha=0.2)
    #if m and m2: plt.fill_between(data.drift.index.values[f1:f2+1], jw_hpd1_2[f1:f2+1], jw_hpd2_2[f1:f2+1], where=(data.m_probs[aoi,f1:f2+1,2]>0.5), color="C3", alpha=0.2)

    #if m1 and not m: plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1], s=10, color="C0")
    #if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean_1[f1:f2+1], s=10, color="C2")
    #if m2 and not m: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean_2[f1:f2+1], s=10, color="C3")
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fw_mean[f1:f2+1], s=10, color="C0")
    #if m and m1: plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1], s=10, color=m_colors[f1:f2+1])
    #if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean_1[f1:f2+1], s=10, color=m_colors_1[f1:f2+1])
    #if m and m2: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean_2[f1:f2+1], s=10, color=m_colors_2[f1:f2+1])
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fw_mean[f1:f2+1], s=10, color="C0")
    #if binder: plt.scatter([data.drift.index.values[f1], data.drift.index.values[f2]], [w_mean, w_mean], s=10, color="C0")
    #if junk: plt.scatter([data.drift.index.values[f1], data.drift.index.values[f2]], [jw_mean, jw_mean], s=10, color="C0")
    #if binder: plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1], s=10, color=z_colors[f1:f2+1])
    #if junk: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean[f1:f2+1], s=10, color=j_colors[f1:f2+1])
    plt.ylim(0,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("width", fontsize=20)
    
    # background 
    plt.subplot(5,1,5)
    plt.fill_between(data.drift.index.values[f1:f2+1], b_hpd1[f1:f2+1], b_hpd2[f1:f2+1], color="C0", alpha=0.2)
    plt.scatter(data.drift.index.values[f1:f2+1], b_mean[f1:f2+1], s=10, color="C0")
    plt.ylim(0,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("background", fontsize=20)
    plt.xlabel("frame #", fontsize=20)
    
    #plt.legend()
    plt.tight_layout()
    plt.show()

def view_feature_summary(aoi, data, f1, f2, jfeature, feature, binder, junk):
    
    #b_hpd1, b_hpd2 = hpd(dist.Gamma(data.b_loc[aoi] * data.b_beta[aoi], data.b_beta[aoi]).sample((1000,)))
    #b_hpd1, b_hpd2 = hpd(dist.Gamma(data.b_loc[aoi,:,0,0] * data.b_beta, data.b_beta).sample((1000,)))
    #b_mean = data.b_loc[aoi,:,0,0].cpu().numpy()

    if feature:
        #b_hpd1, b_hpd2 = hpd(dist.Gamma(data.b_loc_1[aoi,:,0,0] * data.b_beta_1, data.b_beta_1).sample((1000,)))
        #b_mean = data.b_loc_1[aoi,:,0,0].cpu().numpy()

        h_hpd1, h_hpd2 = hpd(dist.Gamma(data.h_loc[aoi,:,0,0] * data.h_beta[aoi,:,0,0], data.h_beta[aoi,:,0,0]).sample((1000,)))
        h_mean = data.h_loc[aoi,:,0,0].cpu().numpy()

        x_hpd1, x_hpd2 = hpd(Location(data.x_mode[aoi,:,0,0], data.size[aoi,:,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        x_mean = data.x_mode[aoi,:,0,0].cpu().numpy()

        y_hpd1, y_hpd2 = hpd(Location(data.y_mode[aoi,:,0,0], data.size[aoi,:,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        y_mean = data.y_mode[aoi,:,0,0].cpu().numpy()

        #w_hpd1, w_hpd2 = hpd(dist.Gamma(data.w_loc * data.w_beta * data.size[aoi,:,0,0], data.w_beta * data.size[aoi,:,0,0]).sample((1000,)))
        #w_mean = data.w_loc.cpu().numpy()
        w_hpd1, w_hpd2 = hpd(Location(data.w_mode, data.w_size, 0.5, 3.).sample((1000,)))
        w_mean = data.w_mode.cpu().numpy()

    if jfeature:
        #b_hpd1, b_hpd2 = hpd(dist.Gamma(data.b_loc_2[aoi,:,0,0] * data.b_beta_2, data.b_beta_2).sample((1000,)))
        #b_mean = data.b_loc_2[aoi,:,0,0].cpu().numpy()

        jh_hpd1_1, jh_hpd2_1 = hpd(dist.Gamma(data.h_loc_2[1,aoi,:,0,0] * data.h_beta_2[1,aoi,:,0,0], data.h_beta_2[1,aoi,:,0,0]).sample((1000,)))
        jh_mean_1 = data.h_loc_2[1,aoi,:,0,0].cpu().numpy()

        jx_hpd1_1, jx_hpd2_1 = hpd(Location(data.x_mode_2[1,aoi,:,0,0], data.size_2[1,aoi,:,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jx_mean_1 = data.x_mode_2[1,aoi,:,0,0].cpu().numpy()

        jy_hpd1_1, jy_hpd2_1 = hpd(Location(data.y_mode_2[1,aoi,:,0,0], data.size_2[1,aoi,:,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jy_mean_1 = data.y_mode_2[1,aoi,:,0,0].cpu().numpy()

        jw_hpd1_1, jw_hpd2_1 = hpd(dist.Gamma(data.w_loc_2 * data.w_beta_2 * data.size_2[1,aoi,:,0,0], data.w_beta_2 * data.size_2[1,aoi,:,0,0]).sample((1000,)))
        jw_mean_1 = data.w_loc_2.cpu().numpy()

        jh_hpd1_2, jh_hpd2_2 = hpd(dist.Gamma(data.h_loc_2[0,aoi,:,0,0] * data.h_beta_2[0,aoi,:,0,0], data.h_beta_2[0,aoi,:,0,0]).sample((1000,)))
        jh_mean_2 = data.h_loc_2[0,aoi,:,0,0].cpu().numpy()

        jx_hpd1_2, jx_hpd2_2 = hpd(Location(data.x_mode_2[0,aoi,:,0,0], data.size_2[0,aoi,:,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jx_mean_2 = data.x_mode_2[0,aoi,:,0,0].cpu().numpy()

        jy_hpd1_2, jy_hpd2_2 = hpd(Location(data.y_mode_2[0,aoi,:,0,0], data.size_2[0,aoi,:,0,0], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jy_mean_2 = data.y_mode_2[0,aoi,:,0,0].cpu().numpy()

        jw_hpd1_2, jw_hpd2_2 = hpd(dist.Gamma(data.w_loc_2 * data.w_beta_2 * data.size_2[0,aoi,:,0,0], data.w_beta_2 * data.size_2[0,aoi,:,0,0]).sample((1000,)))
        jw_mean_2 = data.w_loc_2.cpu().numpy()


    if binder:
        z_colors = np.zeros((len(data.drift),4)) + to_rgba_array("C2")
        z_colors[:,3] = data.z_probs.cpu().numpy()[aoi,:,1]

        #h_hpd1, h_hpd2 = hpd(dist.Gamma(data.h_loc * data.size[aoi] * data.h_beta[aoi], data.h_beta[aoi]).sample((1000,)))
        #h_mean = (data.h_loc * data.size[aoi]).cpu().numpy()

        x_hpd1, x_hpd2 = hpd(Location(data.x_mode[aoi], data.x_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        x_mean = data.x_mode[aoi].cpu().numpy()

        y_hpd1, y_hpd2 = hpd(Location(data.y_mode[aoi], data.y_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        y_mean = data.y_mode[aoi].cpu().numpy()

        #w_hpd1, w_hpd2 = hpd(dist.Gamma(data.w_loc[aoi] * data.w_beta * data.size[aoi], data.w_beta * data.size[aoi]).sample((1000,)))
        w_mean = data.w_loc[aoi].cpu().numpy()

        h_hpd1, h_hpd2 = hpd(dist.Gamma(data.h_loc[aoi] * data.h_beta[aoi], data.h_beta[aoi]).sample((1000,)))
        h_mean = data.h_loc[aoi].cpu().numpy()

        #x_hpd1, x_hpd2 = hpd(Location(data.x_mode[aoi], data.x_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #x_mean = data.x_mode[aoi].cpu().numpy()

        #y_hpd1, y_hpd2 = hpd(Location(data.y_mode[aoi], data.y_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #y_mean = data.y_mode[aoi].cpu().numpy()

        w_hpd1, w_hpd2 = hpd(dist.Gamma(data.w_loc[aoi] * data.w_beta[aoi], data.w_beta[aoi]).sample((1000,)))
        #w_mean = data.w_loc[aoi].cpu().numpy()

    if junk:
        j_colors = np.zeros((len(data.drift),4)) + to_rgba_array("C3")
        j_colors[:,3] = data.j_probs.cpu().numpy()[aoi,:,1]

        #jh_hpd1, jh_hpd2 = hpd(dist.Gamma(data.h_loc * data.jsize[aoi] * data.jh_beta[aoi], data.jh_beta[aoi]).sample((1000,)))
        #jh_mean = (data.h_loc * data.jsize[aoi]).cpu().numpy()

        jx_hpd1, jx_hpd2 = hpd(Location(data.jx_mode[aoi], data.jx_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jx_mean = data.jx_mode[aoi].cpu().numpy()

        jy_hpd1, jy_hpd2 = hpd(Location(data.jy_mode[aoi], data.jy_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        jy_mean = data.jy_mode[aoi].cpu().numpy()

        #jw_hpd1, jw_hpd2 = hpd(dist.Gamma(data.jw_loc[aoi] * data.w_beta * data.jsize[aoi], data.w_beta * data.jsize[aoi]).sample((1000,)))
        jw_mean = data.jw_loc[aoi].cpu().numpy()

        jh_hpd1, jh_hpd2 = hpd(dist.Gamma(data.jh_loc[aoi] * data.jh_beta[aoi], data.jh_beta[aoi]).sample((1000,)))
        jh_mean = data.jh_loc[aoi].cpu().numpy()

        #jx_hpd1, jx_hpd2 = hpd(Location(data.jx_mode[aoi], data.jx_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #jx_mean = data.jx_mode[aoi].cpu().numpy()

        #jy_hpd1, jy_hpd2 = hpd(Location(data.jy_mode[aoi], data.jy_size[aoi], -(data.D+3)/2, (data.D+3)).sample((1000,)))
        #jy_mean = data.jy_mode[aoi].cpu().numpy()

        jw_hpd1, jw_hpd2 = hpd(dist.Gamma(data.jw_loc[aoi] * data.jw_beta[aoi], data.jw_beta[aoi]).sample((1000,)))
        #jw_mean = data.jw_loc[aoi].cpu().numpy()


    plt.figure(figsize=(15,15))
    # height
    plt.subplot(5,1,1)
    if feature: plt.fill_between(data.drift.index.values[f1:f2+1], h_hpd1[f1:f2+1], h_hpd2[f1:f2+1], color="C2", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jh_hpd1_1[f1:f2+1], jh_hpd2_1[f1:f2+1], color="C3", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jh_hpd1_2[f1:f2+1], jh_hpd2_2[f1:f2+1], color="C4", alpha=0.2)
    #if feature: plt.fill_between(data.drift.index.values[f1:f2+1], fh_hpd1[f1:f2+1], fh_hpd2[f1:f2+1], color="C0", alpha=0.2)
    if binder: plt.fill_between(data.drift.index.values[f1:f2+1], h_hpd1[f1:f2+1], h_hpd2[f1:f2+1], where=(data.z_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C2", alpha=0.2)
    if junk: plt.fill_between(data.drift.index.values[f1:f2+1], jh_hpd1[f1:f2+1], jh_hpd2[f1:f2+1], where=(data.j_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C3", alpha=0.2)

    if feature: plt.scatter(data.drift.index.values[f1:f2+1], h_mean[f1:f2+1], s=10, color="C2")
    if jfeature: plt.scatter(data.drift.index.values[f1:f2+1], jh_mean_1[f1:f2+1], s=10, color="C3")
    if jfeature: plt.scatter(data.drift.index.values[f1:f2+1], jh_mean_2[f1:f2+1], s=10, color="C4")
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fh_mean[f1:f2+1], s=10, color="C0")
    if binder: plt.scatter(data.drift.index.values[f1:f2+1], h_mean[f1:f2+1], s=10, color=z_colors[f1:f2+1])
    if junk: plt.scatter(data.drift.index.values[f1:f2+1], jh_mean[f1:f2+1], s=10, color=j_colors[f1:f2+1])
    plt.ylim(0,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("height", fontsize=20) 

    # x position
    plt.subplot(5,1,2)
    #plt.hlines([-2.5,0,2.5], data.drift.index.values[0], data.drift.index.values[-1], linestyles='dashed')
    if feature: plt.fill_between(data.drift.index.values[f1:f2+1], x_hpd1[f1:f2+1], x_hpd2[f1:f2+1], color="C2", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jx_hpd1_1[f1:f2+1], jx_hpd2_1[f1:f2+1], color="C3", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jx_hpd1_2[f1:f2+1], jx_hpd2_2[f1:f2+1], color="C4", alpha=0.2)
    #if feature: plt.fill_between(data.drift.index.values[f1:f2+1], fx_hpd1[f1:f2+1], fx_hpd2[f1:f2+1], color="C0", alpha=0.2)
    if binder: plt.fill_between(data.drift.index.values[f1:f2+1], x_hpd1[f1:f2+1], x_hpd2[f1:f2+1], where=(data.z_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C2", alpha=0.2)
    if junk: plt.fill_between(data.drift.index.values[f1:f2+1], jx_hpd1[f1:f2+1], jx_hpd2[f1:f2+1], where=(data.j_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C3", alpha=0.2)

    if feature: plt.scatter(data.drift.index.values[f1:f2+1], x_mean[f1:f2+1], s=10, color="C2")
    if jfeature: plt.scatter(data.drift.index.values[f1:f2+1], jx_mean_1[f1:f2+1], s=10, color="C3")
    if jfeature: plt.scatter(data.drift.index.values[f1:f2+1], jx_mean_2[f1:f2+1], s=10, color="C4")
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fx_mean[f1:f2+1], s=10, color="C0")
    if binder: plt.scatter(data.drift.index.values[f1:f2+1], x_mean[f1:f2+1], s=10, color=z_colors[f1:f2+1])
    if junk: plt.scatter(data.drift.index.values[f1:f2+1], jx_mean[f1:f2+1], s=10, color=j_colors[f1:f2+1])
    plt.ylim(-(data.D+3)/2, (data.D+3)/2)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("x position", fontsize=20)
    
    # y position
    plt.subplot(5,1,3)
    #plt.hlines([-2.5,0,2.5], data.drift.index.values[0], data.drift.index.values[-1], linestyles='dashed')
    if feature: plt.fill_between(data.drift.index.values[f1:f2+1], y_hpd1[f1:f2+1], y_hpd2[f1:f2+1], color="C2", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jy_hpd1_1[f1:f2+1], jy_hpd2_1[f1:f2+1], color="C3", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jy_hpd1_2[f1:f2+1], jy_hpd2_2[f1:f2+1], color="C4", alpha=0.2)
    #if feature: plt.fill_between(data.drift.index.values[f1:f2+1], fy_hpd1[f1:f2+1], fy_hpd2[f1:f2+1], color="C0", alpha=0.2)
    if binder: plt.fill_between(data.drift.index.values[f1:f2+1], y_hpd1[f1:f2+1], y_hpd2[f1:f2+1], where=(data.z_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C2", alpha=0.2)
    if junk: plt.fill_between(data.drift.index.values[f1:f2+1], jy_hpd1[f1:f2+1], jy_hpd2[f1:f2+1], where=(data.j_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C3", alpha=0.2)

    if feature: plt.scatter(data.drift.index.values[f1:f2+1], y_mean[f1:f2+1], s=10, color="C2")
    if jfeature: plt.scatter(data.drift.index.values[f1:f2+1], jy_mean_1[f1:f2+1], s=10, color="C3")
    if jfeature: plt.scatter(data.drift.index.values[f1:f2+1], jy_mean_2[f1:f2+1], s=10, color="C4")
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fy_mean[f1:f2+1], s=10, color="C0")
    if binder: plt.scatter(data.drift.index.values[f1:f2+1], y_mean[f1:f2+1], s=10, color=z_colors[f1:f2+1])
    if junk: plt.scatter(data.drift.index.values[f1:f2+1], jy_mean[f1:f2+1], s=10, color=j_colors[f1:f2+1])
    plt.ylim(-(data.D+3)/2, (data.D+3)/2)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("y position", fontsize=20)
    
    # width
    plt.subplot(5,1,4)
    if feature: plt.fill_between(data.drift.index.values[f1:f2+1], w_hpd1[f1:f2+1], w_hpd2[f1:f2+1], color="C2", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jw_hpd1_1[f1:f2+1], jw_hpd2_1[f1:f2+1], color="C3", alpha=0.2)
    if jfeature: plt.fill_between(data.drift.index.values[f1:f2+1], jw_hpd1_2[f1:f2+1], jw_hpd2_2[f1:f2+1], color="C4", alpha=0.2)
    #if feature: plt.fill_between(data.drift.index.values[f1:f2+1], fw_hpd1[f1:f2+1], fw_hpd2[f1:f2+1], color="C0", alpha=0.2)
    if binder: plt.fill_between(data.drift.index.values[f1:f2+1], w_hpd1[f1:f2+1], w_hpd2[f1:f2+1], where=(data.z_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C2", alpha=0.2)
    if junk: plt.fill_between(data.drift.index.values[f1:f2+1], jw_hpd1[f1:f2+1], jw_hpd2[f1:f2+1], where=(data.j_probs[aoi,f1:f2+1,1].numpy()>0.5), color="C3", alpha=0.2)

    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1], s=10, color="C2")
    #if jfeature: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean[f1:f2+1], s=10, color="C3")
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fw_mean[f1:f2+1], s=10, color="C0")
    if binder: plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1], s=10, color=z_colors[f1:f2+1])
    if junk: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean[f1:f2+1], s=10, color=j_colors[f1:f2+1])
    #if feature: plt.scatter(data.drift.index.values[f1:f2+1], fw_mean[f1:f2+1], s=10, color="C0")
    #if binder: plt.scatter([data.drift.index.values[f1], data.drift.index.values[f2]], [w_mean, w_mean], s=10, color="C0")
    #if junk: plt.scatter([data.drift.index.values[f1], data.drift.index.values[f2]], [jw_mean, jw_mean], s=10, color="C0")
    #if binder: plt.scatter(data.drift.index.values[f1:f2+1], w_mean[f1:f2+1], s=10, color=z_colors[f1:f2+1])
    #if junk: plt.scatter(data.drift.index.values[f1:f2+1], jw_mean[f1:f2+1], s=10, color=j_colors[f1:f2+1])
    plt.ylim(0,)
    plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    plt.ylabel("width", fontsize=20)
    
    # background 
    #plt.subplot(5,1,5)
    #plt.fill_between(data.drift.index.values[f1:f2+1], b_hpd1[f1:f2+1], b_hpd2[f1:f2+1], color="C0", alpha=0.2)
    #plt.scatter(data.drift.index.values[f1:f2+1], b_mean[f1:f2+1], s=10, color="C0")
    #plt.ylim(0,)
    #plt.xlim(data.drift.index.values[f1:f2+1].min()-2, data.drift.index.values[f1:f2+1].max()+2)
    #plt.ylabel("background", fontsize=20)
    #plt.xlabel("frame #", fontsize=20)
    
    #plt.legend()
    plt.tight_layout()
    plt.show()
    
def view_aoi(aoi, frame, data, target, binder, junk):
    #fig = plt.figure(figsize=(30,1.5), dpi=600)
    fig = plt.figure(figsize=(15,3), dpi=600)
    for i in range(20):
        ax = fig.add_subplot(2,10,i+1)
        plt.title("f #{:d}".format(data.drift.index[frame+i]), fontsize=15)
        plt.imshow(data._store[aoi,frame+i].cpu(), cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
        if target:
            plt.plot(data.target.iloc[aoi, 2] + data.drift.iloc[frame+i, 1] + 0.5, data.target.iloc[aoi, 1] + data.drift.iloc[frame+i, 0] + 0.5, "b+", markersize=10, mew=3, alpha=0.7)
        if binder:
            z_color = to_rgba_array("C2", data.z_probs.cpu().numpy()[aoi,frame+i,1])[0]
            plt.gca().add_patch(Rectangle((0, 0), data.D*data.z_probs.cpu().numpy()[aoi,frame+i,1], 0.25, edgecolor=z_color, lw=4, facecolor="none"))
            #plt.plot(data.target.iloc[aoi, 2] + data.y_mode[aoi, frame+i] + 0.5, data.target.iloc[aoi, 1] + data.x_mode[aoi, frame+i] + 0.5, "g+", markersize=10, mew=3, alpha=data.z_probs[aoi, frame+i, 1])
        if junk:
            j_color = to_rgba_array("C3", data.j_probs.cpu().numpy()[aoi,frame+i,1])[0]
            plt.gca().add_patch(Rectangle((0, data.D-1), data.D*data.j_probs.cpu().numpy()[aoi,frame+i,1], 0.25, edgecolor=j_color, lw=4, facecolor="none"))
            #plt.plot(data.target.iloc[aoi, 2] + data.jy_mode[aoi, frame+i] + 0.5, data.target.iloc[aoi, 1] + data.jx_mode[aoi, frame+i] + 0.5, "r+", markersize=10, mew=3, alpha=data.j_probs[aoi, frame+i, 1])
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

def hpd(trace, mass_frac=0.95) :
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace), axis=0)
    #print(d)

    # Number of total samples taken
    n = len(trace)
    f = trace.shape[1]
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width, axis=0)
    
    # Return interval
    return np.array([d[min_int, np.arange(f)], d[min_int+n_samples, np.arange(f)]])


def Location(mode, size, loc, scale):
    mode = (mode - loc) / scale
    concentration1 = mode * (size - 2) + 1
    concentration0 = (1 - mode) * (size - 2) + 1
    base_distribution = dist.Beta(concentration1, concentration0)
    transforms =  [AffineTransform(loc=loc, scale=scale)]
    return dist.TransformedDistribution(base_distribution, transforms)
