import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pyro
from pyro import param
from pyro.ops.stats import hpdi
from matplotlib.colors import to_rgba_array
import torch.distributions as dist
from cosmos.models.model import GaussianSpot
from cosmos.models.helper import z_probs_calc, k_probs_calc, ScaledBeta


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


def view_m_probs(aoi, data, f1, f2, m, z, labels, prefix):
    if m:
        k_probs = k_probs_calc(
            param(f"{prefix}/m_probs")[aoi]).squeeze()

    if z:
        z_probs = z_probs_calc(
            pyro.param("d/m_probs")[aoi],
            pyro.param("d/theta_probs")[aoi]).squeeze()

    plt.figure(figsize=(25, 5))
    if m:
        for k in range(2):
            plt.plot(
                data.drift.index.values[f1:f2+1], k_probs[f1:f2+1, k],
                marker="o", ms=5, color="C{}".format(k), label="m{}".format(k))
    if z:
        plt.plot(
            data.drift.index.values[f1:f2+1], z_probs[f1:f2+1],
            marker="o", ms=5, color="C2", label="z")
    if labels:
        plt.plot(
            data.drift.index.values[f1:f2+1],
            data.labels.iloc[aoi*data.F+f1:aoi*data.F+f2+1, 0],
            marker="o", ms=5, color="C3", label="spotpicker")
    plt.ylim(-0.02,)
    plt.xlim(
        data.drift.index.values[f1:f2+1].min()-0.1,
        data.drift.index.values[f1:f2+1].max()+0.1)
    plt.ylabel("probability", fontsize=30)
    plt.xlabel("frame #", fontsize=30)
    plt.gca().set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.gca().tick_params(axis="x", labelsize=25)
    plt.gca().tick_params(axis="y", labelsize=25)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.show()


def view_parameters(aoi, data, f1, f2, m, params, prefix, theta):
    if m:
        k_probs = k_probs_calc(
            param(f"{prefix}/m_probs")[aoi]).squeeze()
        m_colors = np.zeros((2, len(data.drift), 4))
        m_colors[0] += to_rgba_array("C0")
        m_colors[0, :, 3] = k_probs[:, 0]
        m_colors[1] += to_rgba_array("C1")
        m_colors[1, :, 3] = k_probs[:, 1]

    plt.figure(figsize=(15, 3 * len(params)))
    for i, p in enumerate(params):
        plt.subplot(5, 1, i+1)
        if p == "background":
            hpd = hpdi(dist.Gamma(
                param(f"{prefix}/b_loc").data[aoi]
                * param(f"{prefix}/b_beta").data[aoi],
                param(f"{prefix}/b_beta").data[aoi])
                .sample((500,)), 0.95, dim=0)
            mean = param(f"{prefix}/b_loc").data[aoi]
            plt.ylim(0, hpd.max()+1)
        elif p == "intensity":
            hpd = hpdi(dist.Gamma(
                param(f"{prefix}/h_loc").data[aoi]
                * param(f"{prefix}/h_beta").data[aoi],
                param(f"{prefix}/h_beta").data[aoi])
                .sample((500,)), 0.95, dim=0)
            mean = param(f"{prefix}/h_loc").data[aoi]
            plt.ylim(0, hpd.max()+10)
        elif p == "x":
            hpd = hpdi(ScaledBeta(
                param(f"{prefix}/x_mode").data[aoi],
                param(f"{prefix}/size").data[aoi],
                -(data.D+3)/2, data.D+3)
                .sample((500,)), 0.95, dim=0)
            hpd = hpd * (data.D+3) - (data.D+3)/2
            mean = param(f"{prefix}/x_mode").data[aoi]
            plt.ylim(-(data.D+3)/2, (data.D+3)/2)
        elif p == "y":
            hpd = hpdi(ScaledBeta(
                param(f"{prefix}/y_mode").data[aoi],
                param(f"{prefix}/size").data[aoi],
                -(data.D+3)/2, data.D+3)
                .sample((500,)), 0.95, dim=0)
            hpd = hpd * (data.D+3) - (data.D+3)/2
            mean = param(f"{prefix}/y_mode").data[aoi]
            plt.ylim(-(data.D+3)/2, (data.D+3)/2)
        elif p == "width":
            hpd = hpdi(ScaledBeta(
                param(f"{prefix}/w_mode").data[aoi],
                param(f"{prefix}/w_size").data[aoi], 0.5, 2.5)
                .sample((500,)), 0.95, dim=0)
            hpd = hpd * 2.5 + 0.5
            mean = param(f"{prefix}/w_mode").data[aoi]

        hpd = hpd.squeeze()
        mean = mean.squeeze()
        if p == "background":
            plt.fill_between(
                data.drift.index.values[f1:f2+1],
                hpd[0][f1:f2+1], hpd[1][f1:f2+1], color="C0", alpha=0.2)
            plt.scatter(
                data.drift.index.values[f1:f2+1], mean[f1:f2+1],
                s=10, color="C0", label="K")
        else:
            for k in range(2):
                plt.fill_between(
                    data.drift.index.values[f1:f2+1],
                    hpd[0][f1:f2+1, k], hpd[1][f1:f2+1, k],
                    where=(k_probs[f1:f2+1, k] > 0.5) if m else None,
                    color="C{}".format(k), alpha=0.2)
                plt.scatter(
                    data.drift.index.values[f1:f2+1], mean[f1:f2+1, k],
                    s=10, color=m_colors[k] if m else "C{}".format(k),
                    label="K={}".format(k))
        plt.xlim(data.drift.index.values[f1:f2+1].min()-2,
                 data.drift.index.values[f1:f2+1].max()+2)
        plt.ylabel(p, fontsize=20)

    plt.xlabel("frame #", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.show()


def view_aoi(aoi, frame, data, target, z, labels, prefix):
    if z:
        z_probs = z_probs_calc(
            pyro.param("d/m_probs")[aoi],
            pyro.param("d/theta_probs")[aoi]).squeeze()

    m_mask = k_probs_calc(
        param(f"{prefix}/m_probs")[aoi]) > 0.5
    ideal_spot = GaussianSpot(data, 2)
    ideal_data = ideal_spot(
                    aoi, m_mask, param("{}/h_loc".format(prefix))[aoi],
                    param("{}/w_mode".format(prefix))[aoi],
                    param("{}/x_mode".format(prefix))[aoi],
                    param("{}/y_mode".format(prefix))[aoi],
                    param("{}/b_loc".format(prefix))[aoi]) + param("offset")

    f, frames = frame
    fig = plt.figure(figsize=(15, 3), dpi=600)
    for i in range(10):
        try:
            fig.add_subplot(2, 10, i+1)
            
            plt.title("f #{:d}".format(data.drift.index[frames[f+i]]), fontsize=15)
            plt.imshow(
                data._store[aoi, frames[f+i]].cpu(),
                cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
            if target:
                plt.plot(
                    data.target.iloc[aoi, 2] + data.drift.iloc[frames[f+i], 1] + 0.5,
                    data.target.iloc[aoi, 1] + data.drift.iloc[frames[f+i], 0] + 0.5,
                    "b+", markersize=10, mew=3, alpha=0.7)
            if z:
                z_color = to_rgba_array("C2", z_probs[frames[f+i]])[0]
                plt.gca().add_patch(
                    Rectangle((0, 0), data.D*z_probs[frames[f+i]], 0.25,
                              edgecolor=z_color, lw=4, facecolor="none"))
            if labels:
                if data.labels.iloc[aoi*data.F+frames[f+i], 0] == 1:
                    plt.gca().add_patch(
                        Rectangle((0, data.D-1), data.D, 0.25,
                                  edgecolor="C3", lw=4, facecolor="none"))
        except:
            pass

    for i in range(10):
        try:
            fig.add_subplot(2, 10, i+11)
            plt.imshow(
                ideal_data.data[frames[f+i], :, :, 0].cpu(),
                cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
            if target:
                plt.plot(
                    data.target.iloc[aoi, 2] + data.drift.iloc[frames[f+i], 1] + 0.5,
                    data.target.iloc[aoi, 1] + data.drift.iloc[frames[f+i], 0] + 0.5,
                    "b+", markersize=10, mew=3, alpha=0.7)
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])
        except:
            pass
    plt.show()
