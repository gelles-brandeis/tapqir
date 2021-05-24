# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyro.ops.stats import pi, quantile
from pyroapi import handlers
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def ci_from_trace(tr, sites, ci=0.95, num_samples=500):
    ci_stats = {}
    for name in sites:
        ci_stats[name] = {}
        hpd = pi(
            tr.nodes[name]["fn"].sample((num_samples,)).data.squeeze().cpu(),
            ci,
            dim=0,
        )
        mean = tr.nodes[name]["fn"].mean.data.squeeze().cpu()
        ci_stats[name]["ul"] = hpd[1]
        ci_stats[name]["ll"] = hpd[0]
        ci_stats[name]["mean"] = mean
    return ci_stats


def save_stats(model, path, num_samples):
    # compute and save theta_samples
    model._compute_theta_samples(num_samples)
    torch.save(model.theta_samples, Path(path) / "theta_samples.pt")

    # global parameters
    data = {}
    # snr
    data["snr"] = model.snr().mean().item()
    # parameters
    guide_tr = handlers.trace(model.guide).get_trace()
    global_params = ["gain", "pi", "lamda", "proximity"]
    ci_stats = ci_from_trace(guide_tr, global_params)
    for param in global_params:
        if param == "pi":
            data[f"{param}_mean"] = ci_stats[param]["mean"][1].item()
            data[f"{param}_ll"] = ci_stats[param]["ll"][1].item()
            data[f"{param}_ul"] = ci_stats[param]["ul"][1].item()
        else:
            data[f"{param}_mean"] = ci_stats[param]["mean"].item()
            data[f"{param}_ll"] = ci_stats[param]["ll"].item()
            data[f"{param}_ul"] = ci_stats[param]["ul"].item()

    # check convergence status
    data["converged"] = False
    for line in open(Path(path) / "run.log"):
        if "model converged" in line:
            data["converged"] = True

    # classification statistics
    if model.data.labels is not None:
        pred_labels = model.z_map.cpu().numpy().ravel()
        true_labels = model.data.labels["z"].ravel()

        with np.errstate(divide="ignore", invalid="ignore"):
            data["MCC"] = matthews_corrcoef(true_labels, pred_labels)
        data["Recall"] = recall_score(true_labels, pred_labels, zero_division=0)
        data["Precision"] = precision_score(true_labels, pred_labels, zero_division=0)

        data["TN"], data["FP"], data["FN"], data["TP"] = confusion_matrix(
            true_labels, pred_labels
        ).ravel()

        mask = torch.from_numpy(model.data.labels["z"])
        samples = torch.masked_select(model.z_marginal.cpu(), mask)
        if len(samples):
            z_ll, z_ul = pi(samples, 0.68)
            data["z_median"] = quantile(samples, 0.5).item()
            data["z_ll"] = z_ll.item()
            data["z_ul"] = z_ul.item()
        else:
            data["z_median"] = 0.0
            data["z_ll"] = 0.0
            data["z_ul"] = 0.0

    pd.Series(data).to_csv(
        Path(path) / "statistics.csv",
    )
