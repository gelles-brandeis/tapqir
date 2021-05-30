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
        samples = tr.nodes[name]["fn"].sample((num_samples,)).data.squeeze().cpu()
        hpd = pi(
            samples,
            ci,
            dim=0,
        )
        mean = tr.nodes[name]["fn"].mean.data.squeeze().cpu()
        ci_stats[name]["ul"] = hpd[1]
        ci_stats[name]["ll"] = hpd[0]
        ci_stats[name]["mean"] = mean

        # calculate Keq
        if name == "pi":
            hpd = pi(samples[:, 1] / (1 - samples[:, 1]), ci, dim=0)
            mean = (samples[:, 1] / (1 - samples[:, 1])).mean()
            ci_stats["Keq"] = {}
            ci_stats["Keq"]["ul"] = hpd[1]
            ci_stats["Keq"]["ll"] = hpd[0]
            ci_stats["Keq"]["mean"] = mean

    return ci_stats


def save_stats(model, path):
    # change device to cpu
    model.to("cpu")
    model.batch_size = model.n = None
    # global parameters
    global_params = ["gain", "pi", "lamda", "proximity"]
    data = pd.DataFrame(index=global_params, columns=["Mean", "95% LL", "95% UL"])
    # parameters
    guide_tr = handlers.trace(model.guide).get_trace()
    local_params = [
        "d/height_0",
        "d/height_1",
        "d/width_0",
        "d/width_1",
        "d/x_0",
        "d/x_1",
        "d/y_0",
        "d/y_1",
        "d/background",
    ]
    ci_stats = ci_from_trace(guide_tr, global_params + local_params)
    params_dict = {}
    for param in global_params + ["Keq"]:
        if param == "pi":
            data.loc[param, "Mean"] = params_dict[f"{param}_mean"] = ci_stats[param][
                "mean"
            ][1].item()
            data.loc[param, "95% LL"] = params_dict[f"{param}_ll"] = ci_stats[param][
                "ll"
            ][1].item()
            data.loc[param, "95% UL"] = params_dict[f"{param}_ul"] = ci_stats[param][
                "ul"
            ][1].item()
        else:
            data.loc[param, "Mean"] = params_dict[f"{param}_mean"] = ci_stats[param][
                "mean"
            ].item()
            data.loc[param, "95% LL"] = params_dict[f"{param}_ll"] = ci_stats[param][
                "ll"
            ].item()
            data.loc[param, "95% UL"] = params_dict[f"{param}_ul"] = ci_stats[param][
                "ul"
            ].item()
    for param in local_params:
        if param == "d/background":
            params_dict[f"{param}_mean"] = ci_stats[param]["mean"]
            params_dict[f"{param}_ll"] = ci_stats[param]["ll"]
            params_dict[f"{param}_ul"] = ci_stats[param]["ul"]
        elif param.endswith("_0"):
            base_name = param.split("_")[0]
            params_dict[f"{base_name}_mean"] = torch.stack(
                [ci_stats[f"{base_name}_{k}"]["mean"] for k in range(model.K)], dim=0
            )
            params_dict[f"{base_name}_ll"] = torch.stack(
                [ci_stats[f"{base_name}_{k}"]["ll"] for k in range(model.K)], dim=0
            )
            params_dict[f"{base_name}_ul"] = torch.stack(
                [ci_stats[f"{base_name}_{k}"]["ul"] for k in range(model.K)], dim=0
            )
    params_dict["d/m_probs"] = model.m_probs.data
    params_dict["d/z_probs"] = model.z_probs.data
    params_dict["d/j_probs"] = model.j_probs.data
    params_dict["z_marginal"] = model.z_marginal.data
    params_dict["z_map"] = model.z_map.data
    model.params = params_dict

    # snr
    data.loc["SNR", "Mean"] = model.snr().mean().item()
    # check convergence status
    data.loc["trained", "Mean"] = False
    for line in open(model.run_path / "run.log"):
        if "model converged" in line:
            data.loc["trained", "Mean"] = True

    # classification statistics
    if model.data.ontarget.labels is not None:
        pred_labels = model.z_map.cpu().numpy().ravel()
        true_labels = model.data.ontarget.labels["z"].ravel()

        with np.errstate(divide="ignore", invalid="ignore"):
            data.loc["MCC", "Mean"] = matthews_corrcoef(true_labels, pred_labels)
        data.loc["Recall", "Mean"] = recall_score(
            true_labels, pred_labels, zero_division=0
        )
        data.loc["Precision", "Mean"] = precision_score(
            true_labels, pred_labels, zero_division=0
        )

        (
            data.loc["TN", "Mean"],
            data.loc["FP", "Mean"],
            data.loc["FN", "Mean"],
            data.loc["TP", "Mean"],
        ) = confusion_matrix(true_labels, pred_labels, labels=(0, 1)).ravel()

        mask = torch.from_numpy(model.data.ontarget.labels["z"])
        samples = torch.masked_select(model.z_marginal.cpu(), mask)
        if len(samples):
            z_ll, z_ul = pi(samples, 0.68)
            data.loc["p(specific)", "Mean"] = quantile(samples, 0.5).item()
            data.loc["p(specific)", "95% LL"] = z_ll.item()
            data.loc["p(specific)", "95% UL"] = z_ul.item()
        else:
            data.loc["p(specific)", "Mean"] = 0.0
            data.loc["p(specific)", "95% LL"] = 0.0
            data.loc["p(specific)", "95% UL"] = 0.0

    model.statistics = data

    if path is not None:
        path = Path(path)
        torch.save(params_dict, path / "params.tpqr")
        data.to_csv(
            path / "statistics.csv",
        )
