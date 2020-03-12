import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
import pyro
from pyro import param
from pyro.ops.stats import hpdi
from matplotlib.colors import to_rgba_array
import torch.distributions as dist
from cosmos.models.model import GaussianSpot
from cosmos.models.helper import z_probs_calc, k_probs_calc, ScaledBeta, theta_probs_calc, j_probs_calc


def view_glimpse(frame, aoi, aoi_df, drift_df, header,
                 path_glimpse, selected_aoi, all_aois, label, offset):
    height = int(header["height"])
    width = int(header["width"])
    glimpse_number = header["filenumber"][int(frame-1)]
    with open(os.path.join(
            path_glimpse, "{}.glimpse".format(glimpse_number))) as fid:
        fid.seek(header['offset'][int(frame-1)])
        img = np.fromfile(
            fid, dtype='>i2', count=height*width).reshape(height, width)
        img += 2**15
        img.astype(np.uint16)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(img,  cmap='gray',
               vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    if all_aois:
        for j in aoi_df.index.values:
            y_pos = aoi_df.at[j, "abs_y"] + drift_df.at[frame, "abs_dy"] - 5
            x_pos = aoi_df.at[j, "abs_x"] + drift_df.at[frame, "abs_dx"] - 5
            plt.gca().add_patch(
                Rectangle(
                    (y_pos, x_pos), 10, 10, edgecolor="b", facecolor="none"))
            if label:
                plt.gca().text(
                    y_pos, x_pos, str(j), fontsize=10, color="white")
    if selected_aoi:
        y_pos = aoi_df.at[aoi, "abs_y"] + drift_df.at[frame, "abs_dy"] - 5
        x_pos = aoi_df.at[aoi, "abs_x"] + drift_df.at[frame, "abs_dx"] - 5
        plt.gca().add_patch(
            Rectangle((y_pos, x_pos), 10, 10, edgecolor="r", facecolor="none"))

    if offset:
        plt.gca().add_patch(
            Rectangle((0, 0), 70, 70, edgecolor="g", facecolor="none"))
        plt.gca().add_patch(
            Rectangle((0, height-70), 70, 70, edgecolor="g", facecolor="none"))
        plt.gca().add_patch(
            Rectangle((width-70, 0), 70, 70, edgecolor="g", facecolor="none"))
        plt.gca().add_patch(
            Rectangle(
                (width-70, height-70),
                70, 70, edgecolor="g", facecolor="none"))

    plt.show()


def view_m_probs(aoi, data, frames, m, z, sp, labels, predictions, prefix):
    fig, ax = plt.subplots(figsize=(15,3)) 
    #ax.clear()
    n = data.target.index.get_loc(aoi)
    f1 = data.drift.index.get_loc(frames[0])
    f2 = data.drift.index.get_loc(frames[1])
    if m:
        k_probs = k_probs_calc(
            param(f"{prefix}/m_probs")[n]).squeeze()
        for k in range(2):
            ax.plot(
                data.drift.index.values[f1:f2+1], k_probs[f1:f2+1, k],
                marker="o", ms=5, color="C{}".format(k), label="m{}".format(k))

    if z:
        z_probs = z_probs_calc(
            pyro.param("d/m_probs")[n],
            pyro.param("d/theta_probs")[n]).squeeze()
        ax.plot(
            data.drift.index.values[f1:f2+1],
            data.predictions["z_prob"][n, f1:f2+1],
            marker="o", ms=5, color="C2", label="z")

    if predictions:
        ax.plot(
            data.drift.index.values[f1:f2+1],
            data.predictions["z"][n, f1:f2+1],
            marker="o", ms=5, color="C8", label="predictions")

    if labels:
        ax.plot(
            data.drift.index.values[f1:f2+1],
            data.labels["z"][n, f1:f2+1],
            marker="o", ms=5, color="C3", label="label")

    if sp:
        ax.plot(
            data.drift.index.values[f1:f2+1],
            data.labels["spotpicker"][n, f1:f2+1],
            marker="o", ms=5, color="C4", label="spotpicker")

    ax.set_ylim(-0.02,)
    ax.set_xlim(
        data.drift.index.values[f1:f2+1].min()-0.1,
        data.drift.index.values[f1:f2+1].max()+0.1)
    ax.set_ylabel("probability", fontsize=14)
    ax.set_xlabel("frame #", fontsize=14)
    ax.set_title("aoi #{}, n #{}".format(aoi, n), fontsize=14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def view_parameters(aoi, data, frames, m, params, prefix):
    fig, ax = plt.subplots(len(params), 1, sharex=True,
                           figsize=(15, 2.5*len(params))) 
    n = data.target.index.get_loc(aoi)
    f1 = data.drift.index.get_loc(frames[0])
    f2 = data.drift.index.get_loc(frames[1])
    if m:
        k_probs = k_probs_calc(
            param(f"{prefix}/m_probs")[n]).squeeze()
        m_colors = np.zeros((2, len(data.drift), 4))
        m_colors[0] += to_rgba_array("C0")
        m_colors[0, :, 3] = k_probs[:, 0]
        m_colors[1] += to_rgba_array("C1")
        m_colors[1, :, 3] = k_probs[:, 1]

    for i, p in enumerate(params):
        if p == "background":
            hpd = hpdi(dist.Gamma(
                param(f"{prefix}/b_loc").data[n]
                * param(f"{prefix}/b_beta").data[n],
                param(f"{prefix}/b_beta").data[n])
                .sample((500,)), 0.95, dim=0)
            mean = param(f"{prefix}/b_loc").data[n]
            ax[i].set_ylim(0, hpd.max()+1)
        elif p == "intensity":
            hpd = hpdi(dist.Gamma(
                param(f"{prefix}/h_loc").data[n]
                * param(f"{prefix}/h_beta").data[n],
                param(f"{prefix}/h_beta").data[n])
                .sample((500,)), 0.95, dim=0)
            mean = param(f"{prefix}/h_loc").data[n]
            ax[i].set_ylim(0, hpd.max()+10)
        elif p == "x":
            hpd = hpdi(ScaledBeta(
                param(f"{prefix}/x_mode").data[n],
                param(f"{prefix}/size").data[n],
                -(data.D+3)/2, data.D+3)
                .sample((500,)), 0.95, dim=0)
            hpd = hpd * (data.D+3) - (data.D+3)/2
            mean = param(f"{prefix}/x_mode").data[n]
            ax[i].set_ylim(-(data.D+3)/2, (data.D+3)/2)
        elif p == "y":
            hpd = hpdi(ScaledBeta(
                param(f"{prefix}/y_mode").data[n],
                param(f"{prefix}/size").data[n],
                -(data.D+3)/2, data.D+3)
                .sample((500,)), 0.95, dim=0)
            hpd = hpd * (data.D+3) - (data.D+3)/2
            mean = param(f"{prefix}/y_mode").data[n]
            ax[i].set_ylim(-(data.D+3)/2, (data.D+3)/2)
        elif p == "width":
            hpd = hpdi(ScaledBeta(
                param(f"{prefix}/w_mode").data[n],
                param(f"{prefix}/w_size").data[n], 0.5, 2.5)
                .sample((500,)), 0.95, dim=0)
            hpd = hpd * 2.5 + 0.5
            mean = param(f"{prefix}/w_mode").data[n]

        hpd = hpd.squeeze()
        mean = mean.squeeze()
        if p == "background":
            ax[i].fill_between(
                data.drift.index.values[f1:f2+1],
                hpd[0][f1:f2+1], hpd[1][f1:f2+1], color="C0", alpha=0.2)
            ax[i].scatter(
                data.drift.index.values[f1:f2+1], mean[f1:f2+1],
                s=10, color="C0", label="K")
        else:
            for k in range(2):
                ax[i].fill_between(
                    data.drift.index.values[f1:f2+1],
                    hpd[0][f1:f2+1, k], hpd[1][f1:f2+1, k],
                    where=(k_probs[f1:f2+1, k] > 0.5) if m else None,
                    color="C{}".format(k), alpha=0.2)
                ax[i].scatter(
                    data.drift.index.values[f1:f2+1], mean[f1:f2+1, k],
                    s=10, color=m_colors[k] if m else "C{}".format(k),
                    label="K={}".format(k))
        ax[i].set_ylabel(p, fontsize=20)

        ax[i].legend()
    ax[len(params)-1].set_xlim(data.drift.index.values[f1:f2+1].min()-2,
                             data.drift.index.values[f1:f2+1].max()+2)
    ax[len(params)-1].set_xlabel("frame #", fontsize=20)
    plt.tight_layout()
    plt.show()


def view_aoi(data, aoi, frame, z, sp, labels, predictions, target, prefix):
    fig, ax = plt.subplots(2, 10, figsize=(15,3)) 
    n = data.target.index.get_loc(aoi)
    f = data.drift.index.get_loc(frame)
    if z:
        z_probs = z_probs_calc(
            pyro.param("d/m_probs")[n],
            pyro.param("d/theta_probs")[n]).squeeze()

    #m_mask = k_probs_calc(
    #    param(f"{prefix}/m_probs")[n]) > 0.5
    #m_mask = torch.tensor(1).bool()
    m_mask = torch.tensor(data.predictions["m"][n]).reshape(data.F, 1, 1, 2).bool()
    ideal_spot = GaussianSpot(data.target, data.drift, data.D, 2)
    ideal_data = ideal_spot(
                    n, m_mask, param("{}/h_loc".format(prefix))[n],
                    param("{}/w_mode".format(prefix))[n],
                    param("{}/x_mode".format(prefix))[n],
                    param("{}/y_mode".format(prefix))[n],
                    param("{}/b_loc".format(prefix))[n]) + param("offset")

    for i in range(10):
        try:
            ax[0, i].set_title("f #{:d}".format(data.drift.index[f+i]), fontsize=15)
            ax[0, i].imshow(
                data[n, f+i].cpu(),
                cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
            if target:
                ax[0, i].plot(
                    data.target["x"].iloc[n] + data.drift["dx"].iloc[f+i] + 0.5,
                    data.target["y"].iloc[n] + data.drift["dy"].iloc[f+i] + 0.5,
                    "b+", markersize=10, mew=3, alpha=0.7)
            if z:
                z_color = to_rgba_array("C2", z_probs[f+i])[0]
                ax[0, i].add_patch(
                    Rectangle((0, 0), data.D*z_probs[f+i], 0.25,
                              edgecolor=z_color, lw=4, facecolor="none"))
            if predictions:
                if data.predictions["z"][n, f+i] == 1:
                    ax[0][i].add_patch(
                        Rectangle((0, 1), data.D, 0.25,
                                  edgecolor="C8", lw=4, facecolor="none"))
            if labels:
                if data.labels["z"][n, f+i] == 1:
                    ax[0][i].add_patch(
                        Rectangle((0, data.D-1), data.D, 0.25,
                                  edgecolor="C3", lw=4, facecolor="none"))

            if sp:
                if data.labels["spotpicker"][n, f+i] == 1:
                    ax[0][i].add_patch(
                        Rectangle((0, data.D-2), data.D, 0.25,
                                  edgecolor="C4", lw=4, facecolor="none"))
            ax[0, i].axes.get_xaxis().set_ticks([])
            ax[0, i].axes.get_yaxis().set_ticks([])
        except:
            pass

    for i in range(10):
        try:
            ax[1, i].imshow(
                ideal_data.data[f+i, :, :, 0].cpu(),
                cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
            if target:
                ax[1, i].plot(
                    data.target["x"].iloc[n] + data.drift["dx"].iloc[f+i] + 0.5,
                    data.target["y"].iloc[n] + data.drift["dy"].iloc[f+i] + 0.5,
                    "b+", markersize=10, mew=3, alpha=0.7)
            ax[1, i].axes.get_xaxis().set_ticks([])
            ax[1, i].axes.get_yaxis().set_ticks([])
        except:
            pass

    plt.show()


def view_single_aoi(data, aoi, frame, z, sp, labels, target, acc, prefix):
    fig, ax = plt.subplots(1, 2, figsize=(6,3)) 
    n = data.target.index.get_loc(aoi)
    f = data.drift.index.get_loc(frame)
    if z:
        z_probs = z_probs_calc(
            param("d/m_probs")[n, f],
            param("d/theta_probs")[n, f]).squeeze()

    #m_mask = k_probs_calc(
    #    param(f"{prefix}/m_probs")[n,f]) > 0.5
    m_mask = torch.tensor(data.predictions["m"][n]).reshape(data.F, 1, 1, 2).bool()
    ideal_spot = GaussianSpot(data.target, data.drift, data.D, 2)
    ideal_data = ideal_spot(
                    n, m_mask, param("{}/h_loc".format(prefix))[n],
                    param("{}/w_mode".format(prefix))[n],
                    param("{}/x_mode".format(prefix))[n],
                    param("{}/y_mode".format(prefix))[n],
                    param("{}/b_loc".format(prefix))[n]) + param("offset")

    ax[0].set_title("f #{:d}".format(data.drift.index[f]), fontsize=15)
    ax[0].imshow(
        data[n, f].cpu(),
        cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
    if target:
        ax[0].plot(
            data.target["x"].iloc[n] + data.drift["dx"].iloc[f] + 0.5,
            data.target["y"].iloc[n] + data.drift["dy"].iloc[f] + 0.5,
            "b+", markersize=10, mew=3, alpha=0.7)
    if z:
        z_color = to_rgba_array("C2", z_probs)[0]
        ax[0].add_patch(
            Rectangle((0, 0), data.D*z_probs, 0.25,
                      edgecolor=z_color, lw=4, facecolor="none"))
    if labels:
        if data.labels["z"][n, f] == 1:
            ax[0].add_patch(
                Rectangle((0, data.D-1), data.D, 0.25,
                          edgecolor="C3", lw=4, facecolor="none"))
    if sp:
        if data.labels["spotpicker"][n, f] == 1:
            ax[0].add_patch(
                Rectangle((0, data.D-2), data.D, 0.25,
                          edgecolor="C4", lw=4, facecolor="none"))
    ax[0].axes.get_xaxis().set_ticks([])
    ax[0].axes.get_yaxis().set_ticks([])

    ax[1].imshow(
        ideal_data.data[f, :, :, 0].cpu(),
        cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
    if target:
        ax[1].plot(
            data.target["x"].iloc[n] + data.drift["dx"].iloc[f] + 0.5,
            data.target["y"].iloc[n] + data.drift["dy"].iloc[f] + 0.5,
            "b+", markersize=10, mew=3, alpha=0.7)
    ax[1].axes.get_xaxis().set_ticks([])
    ax[1].axes.get_yaxis().set_ticks([])

    plt.show()

def view_globals(z, j, data):

    fig, ax = plt.subplots(2, 2, figsize=(12.5,10))

    if z:
        theta_probs = theta_probs_calc(
            param("d/m_probs"),
            param("d/theta_probs")).squeeze()
        h = ax[0,0].hist(
                param("d/h_loc").squeeze().data.reshape(-1),
                weights=theta_probs.reshape(-1),
                bins=100, label="z", alpha=0.3)
        ax[0,1].hist(
                param("d/w_mode").squeeze().data.reshape(-1),
                weights=theta_probs.reshape(-1),
                bins=100, label="z", alpha=0.3)
        ax[1,0].hist(
                param("d/x_mode").squeeze().data.reshape(-1),
                weights=theta_probs.reshape(-1),
                bins=100, label="z", alpha=0.3)
        ax[1,1].hist(
                param("d/y_mode").squeeze().data.reshape(-1),
                weights=theta_probs.reshape(-1),
                bins=100, label="z", alpha=0.3)
    if j:
        j_probs = j_probs_calc(
            param("d/m_probs"),
            param("d/theta_probs")).squeeze()
        h = ax[0,0].hist(
                param("d/h_loc").squeeze().data.reshape(-1),
                weights=j_probs.reshape(-1),
                bins=100, label="j", alpha=0.3)
        ax[0,1].hist(
                param("d/w_mode").squeeze().data.reshape(-1),
                weights=j_probs.reshape(-1),
                bins=100, label="j", alpha=0.3)
        ax[1,0].hist(
                param("d/x_mode").squeeze().data.reshape(-1),
                weights=j_probs.reshape(-1),
                bins=100, label="j", alpha=0.3)
        ax[1,1].hist(
                param("d/y_mode").squeeze().data.reshape(-1),
                weights=j_probs.reshape(-1),
                bins=100, label="j", alpha=0.3)
    #ax[0,0].hist(
    #        param("d/h_loc").squeeze().data.reshape(-1),
    #        weights=(1 - j_probs.reshape(-1) - theta_probs.reshape(-1)),
    #        bins=100, label="0", alpha=0.3)

    h = torch.linspace(param("d/h_loc").min().item(), param("d/h_loc").max().item(), 100)
    w = torch.linspace(0.5, 3., 100)
    x = torch.linspace(-(data.D+3)/2, (data.D+3)/2, 100)
    ax[0,0].plot(h,
            dist.Gamma(
                param("height_loc").item() * param("height_beta").item(),
                param("height_beta").item()).log_prob(h).exp()
                #* (theta_probs.sum() + j_probs.sum())
                * (h.max() - h.min()) / 100)
    #ax[0,0].plot(h[1],
    #        dist.Gamma(
    #            param("height_loc")[0].item() * param("height_beta")[0].item(),
    #            param("height_beta")[0].item()).log_prob(h[1]).exp()
    #            * (theta_probs.sum() + j_probs.sum())
    #            * (h[1].max() - h[1].min()) / 100
    #            * param("pi_k")[0].item())
    #ax[0,0].plot(h[1],
    #        dist.Gamma(
    #            param("height_loc")[1].item() * param("height_beta")[1].item(),
    #            param("height_beta")[1].item()).log_prob(h[1]).exp()
    #            * (theta_probs.sum() + j_probs.sum())
    #            * (h[1].max() - h[1].min()) / 100
    #            * param("pi_k")[1].item())
    ax[0,1].plot(w,
            ScaledBeta(
                param("width_mode").item(),
                param("width_size").item(), 0.5, 2.5).log_prob(torch.tensor(w - 0.5)/2.5).exp()/2.5
                #* (theta_probs.sum() + j_probs.sum())
                * (w.max() - w.min()) / 100)
    #ax[0,1].plot(w,
    #        ScaledBeta(
    #            param("width_mode")[0].item(),
    #            param("width_size")[0].item(), 0.5, 2.5).log_prob(torch.tensor(w - 0.5)/2.5).exp()/2.5
    #            * (theta_probs.sum() + j_probs.sum())
    #            * (w.max() - w.min()) / 100
    #            * param("pi_k")[0].item())
    #ax[0,1].plot(w,
    #        ScaledBeta(
    #            param("width_mode")[1].item(),
    #            param("width_size")[1].item(), 0.5, 2.5).log_prob(torch.tensor(w - 0.5)/2.5).exp()/2.5
    #            * (theta_probs.sum() + j_probs.sum())
    #            * (w.max() - w.min()) / 100
    #            * param("pi_k")[1].item())
    ax[1,0].plot(x,
            ScaledBeta(
                0.,
                ((data.D+3) / (2*0.5)) ** 2 - 1,
                -(data.D+3)/2, data.D+3).log_prob(torch.tensor(x + (data.D+3)/2)/(data.D+3)).exp()/(data.D+3)
                #* theta_probs.sum()
                * (x.max() - x.min()) / 100)
    ax[1,0].plot(x,
            ScaledBeta(
                0.,
                2.,
                -(data.D+3)/2, data.D+3).log_prob(torch.tensor(x + (data.D+3)/2)/(data.D+3)).exp()/(data.D+3)
                #* j_probs.sum()
                * (x.max() - x.min()) / 100)
    ax[1,1].plot(x,
            ScaledBeta(
                0.,
                ((data.D+3) / (2*0.5)) ** 2 - 1,
                -(data.D+3)/2, data.D+3).log_prob(torch.tensor(x + (data.D+3)/2)/(data.D+3)).exp()/(data.D+3)
                #* theta_probs.sum()
                * (x.max() - x.min()) / 100)
    ax[1,1].plot(x,
            ScaledBeta(
                0.,
                2.,
                -(data.D+3)/2, data.D+3).log_prob(torch.tensor(x + (data.D+3)/2)/(data.D+3)).exp()/(data.D+3)
                #* j_probs.sum()
                * (x.max() - x.min()) / 100)

    #ax[0, 0].set_xlim(0, 7000)
    ax[0, 0].set_xlabel("height", fontsize=20)
    ax[0, 1].set_xlabel("width", fontsize=20)
    ax[1, 0].set_xlabel("x", fontsize=20)
    ax[1, 1].set_xlabel("y", fontsize=20)
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.show()
