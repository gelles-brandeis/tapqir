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


def view_glimpse(frame, aoi, aoi_df, drift_df, header,
                 path_glimpse, selected_aoi, all_aois, label, offset):
    height = int(header["height"])
    width = int(header["width"])
    glimpse_number = header["filenumber"][int(frame-1)]
    with open(os.path.join(path_glimpse, "{}.glimpse".format(glimpse_number))) as fid:
        fid.seek(header['offset'][int(frame-1)])
        img = np.fromfile(fid, dtype='>i2', count=height*width).reshape(height, width)
        img += 2**15
        img.astype(np.uint16)
    plt.figure(figsize=(15, 7.5))
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
            Rectangle((width-70, height-70), 70, 70, edgecolor="g", facecolor="none"))
        # np.stack((img[:70,:70], img[:70,-70:], img[-70:,:70], img[-70:,-70:])).reshape(-1)

    plt.show()


def view_m_probs(aoi, data, f1, f2, m, z, labels, prefix):
    if m:
        k_probs = np.zeros((len(data.drift), 2))
        k_probs[:, 0] = param("{}/m_probs".format(prefix)).squeeze().data[aoi, :, 2] \
            + param("{}/m_probs".format(prefix)).squeeze().data[aoi, :, 3]
        k_probs[:, 1] = param("{}/m_probs".format(prefix)).squeeze().data[aoi, :, 1] \
            + param("{}/m_probs".format(prefix)).squeeze().data[aoi, :, 3]
    if z:
        z_probs = pyro.param("d/z_probs").squeeze().data[aoi, ..., 1]

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
        k_probs[..., 0] = param("{}/m_probs".format(prefix)).data[aoi, ..., 2] \
            + param("{}/m_probs".format(prefix)).data[aoi, ..., 3]
        k_probs[..., 1] = param("{}/m_probs".format(prefix)).data[aoi, ..., 1] \
            + param("{}/m_probs".format(prefix)).data[aoi, ..., 3]
        k_probs = k_probs.squeeze().numpy()
        m_colors = np.zeros((2, len(data.drift), 4))
        m_colors[0] += to_rgba_array("C0")
        m_colors[0, :, 3] = k_probs[:, 0]
        m_colors[1] += to_rgba_array("C1")
        m_colors[1, :, 3] = k_probs[:, 1]
    k_probs = torch.zeros(len(data.drift), 1, 1, 1, 2)

    plt.figure(figsize=(15, 3 * len(params)))
    for i, p in enumerate(params):
        plt.subplot(5, 1, i+1)
        if p == "background":
            hpd = hpdi(dist.Gamma(
                param("{}/b_loc".format(prefix))[aoi].data
                * param("{}/b_beta".format(prefix))[aoi].data,
                param("{}/b_beta".format(prefix))[aoi].data)
                .sample((500,)), 0.95, dim=0)
            mean = param("{}/b_loc".format(prefix))[aoi].data
            plt.ylim(0, hpd.max()+1)
        elif p == "intensity":
            hpd = hpdi(dist.Gamma(
                param("{}/h_loc".format(prefix))[aoi].data
                * param("{}/h_beta".format(prefix))[aoi].data,
                param("{}/h_beta".format(prefix))[aoi].data)
                .sample((500,)), 0.95, dim=0)
            mean = param("{}/h_loc".format(prefix))[aoi].data
            plt.ylim(0, hpd.max()+10)
        elif p == "x":
            x_mode = param("{}/x_mode".format(prefix)).squeeze().data[aoi] \
                / (data.D+3) + 0.5
            size = param("{}/size".format(prefix)).squeeze().data[aoi]
            x_c1 = x_mode * size
            x_c0 = (1 - x_mode) * size
            hpd = (hpdi(dist.Beta(x_c1, x_c0).sample((500,)), 0.95, dim=0) - 0.5) * (data.D+3)
            mean = param("{}/x_mode".format(prefix)).squeeze().data[aoi]
            plt.ylim(-(data.D+3)/2, (data.D+3)/2)
        elif p == "y":
            y_mode = param("{}/y_mode".format(prefix)).squeeze().data[aoi] \
                / (data.D+3) + 0.5
            size = param("{}/size".format(prefix)).squeeze().data[aoi]
            y_c1 = y_mode * size
            y_c0 = (1 - y_mode) * size
            hpd = (hpdi(
                dist.Beta(y_c1, y_c0).sample((500,)), 0.95, dim=0) - 0.5) \
                * (data.D+3)
            mean = param("{}/y_mode".format(prefix)).squeeze().data[aoi]
            plt.ylim(-(data.D+3)/2, (data.D+3)/2)
        elif p == "width":
            w_mode = (param("{}/w_mode".format(prefix))
                      .squeeze().data[aoi] - 0.75) / 1.5
            w_size = param("{}/w_size".format(prefix)).squeeze().data[aoi]
            w_c1 = w_mode * w_size
            w_c0 = (1 - w_mode) * w_size
            hpd = hpdi(
                dist.Beta(w_c1, w_c0).sample((500,)), 0.95, dim=0) * 1.5 + 0.75
            mean = param("{}/w_mode".format(prefix)).squeeze().data[aoi]

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
                    where=(k_probs[f1:f2+1, k]>0.5) if m else None,
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


def view_aoi(aoi, frame, data, target, z, m1, m2, labels, prefix):
    if m1 or m2 or z:
        k_probs = np.zeros((len(data.drift), 2))
        k_probs[:, 0] = param(
            "{}/m_probs".format(prefix)).squeeze().data[aoi, :, 1] \
            + param("{}/m_probs".format(prefix)).squeeze().data[aoi, :, 3]
        k_probs[:, 1] = param(
            "{}/m_probs".format(prefix)).squeeze().data[aoi, :, 2] \
            + param("{}/m_probs".format(prefix)).squeeze().data[aoi, :, 3]
    if z:
        z_probs = (
            pyro.param("d/m_probs").squeeze()[aoi]
            * pyro.param("d/theta_probs").squeeze()[aoi, ..., 1:].sum(dim=-1))\
            .sum(dim=-1).data

    fig = plt.figure(figsize=(15, 3), dpi=600)
    for i in range(20):
        fig.add_subplot(2, 10, i+1)
        plt.title("f #{:d}".format(data.drift.index[frame+i]), fontsize=15)
        plt.imshow(
            data._store[aoi, frame+i].cpu(),
            cmap="gray", vmin=data.vmin, vmax=data.vmax+100)
        if target:
            plt.plot(
                data.target.iloc[aoi, 2] + data.drift.iloc[frame+i, 1] + 0.5,
                data.target.iloc[aoi, 1] + data.drift.iloc[frame+i, 0] + 0.5,
                "b+", markersize=10, mew=3, alpha=0.7)
        if m1:
            plt.plot(
                data.target.iloc[aoi, 2] + data.drift.iloc[frame+i, 1] + 0.5
                + param("{}/y_mode".format(prefix)).squeeze().data[aoi, frame+i, 0],
                data.target.iloc[aoi, 1] + data.drift.iloc[frame+i, 0] + 0.5
                + param("{}/x_mode".format(prefix)).squeeze().data[aoi, frame+i, 0],
                "C0+", markersize=10, mew=3, alpha=k_probs[frame+i, 0])
        if m2:
            plt.plot(
                data.target.iloc[aoi, 2] + data.drift.iloc[frame+i, 1]
                + 0.5 + param("{}/y_mode".format(prefix)).squeeze().data[aoi, frame+i, 1],
                data.target.iloc[aoi, 1] + data.drift.iloc[frame+i, 0]
                + 0.5 + param("{}/x_mode".format(prefix)).squeeze().data[aoi, frame+i, 1],
                "C1+", markersize=10, mew=3, alpha=k_probs[frame+i, 1])
        if z:
            z_color = to_rgba_array("C2", z_probs[frame+i])[0]
            plt.gca().add_patch(
                Rectangle((0, 0), data.D*z_probs[frame+i], 0.25,
                          edgecolor=z_color, lw=4, facecolor="none"))
        if labels:
            if data.labels.iloc[aoi*data.F+frame+i, 0] == 1:
                plt.gca().add_patch(
                    Rectangle((0, data.D-1), data.D, 0.25,
                              edgecolor="C3", lw=4, facecolor="none"))
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()
