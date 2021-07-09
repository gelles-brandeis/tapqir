# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyro.ops.stats import pi, quantile
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from tapqir.handlers import StatsMessenger


def save_stats(model, path, CI=0.95, save_matlab=False):
    # change device to cpu
    model.to("cpu")
    model.batch_size = model.n = None
    # global parameters
    global_params = model._global_params
    data = pd.DataFrame(index=global_params, columns=["Mean", "95% LL", "95% UL"])
    # local parameters
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
    with StatsMessenger(global_params + local_params, CI=CI) as ci_stats:
        model.guide()

    for param in global_params:
        if param == "pi":
            data.loc[param, "Mean"] = ci_stats[param]["Mean"][1].item()
            data.loc[param, "95% LL"] = ci_stats[param]["LL"][1].item()
            data.loc[param, "95% UL"] = ci_stats[param]["UL"][1].item()
        else:
            data.loc[param, "Mean"] = ci_stats[param]["Mean"].item()
            data.loc[param, "95% LL"] = ci_stats[param]["LL"].item()
            data.loc[param, "95% UL"] = ci_stats[param]["UL"].item()
    if model.pspecific is not None:
        ci_stats["d/m_probs"] = model.m_probs.data.cpu()
        ci_stats["d/z_probs"] = model.z_probs.data.cpu()
        ci_stats["d/j_probs"] = model.j_probs.data.cpu()
        ci_stats["p(specific)"] = model.pspecific.data.cpu()
        ci_stats["z_map"] = model.z_map.data.cpu()
    # combine K local params
    for param in local_params:
        if param.endswith("_0"):
            base_name = param.split("_")[0]
            ci_stats[base_name] = {}
            ci_stats[base_name]["Mean"] = torch.stack(
                [ci_stats[f"{base_name}_{k}"]["Mean"] for k in range(model.K)], dim=0
            )
            ci_stats[base_name]["LL"] = torch.stack(
                [ci_stats[f"{base_name}_{k}"]["LL"] for k in range(model.K)], dim=0
            )
            ci_stats[base_name]["UL"] = torch.stack(
                [ci_stats[f"{base_name}_{k}"]["UL"] for k in range(model.K)], dim=0
            )
            for k in range(model.K):
                del ci_stats[f"{base_name}_{k}"]
    model.params = ci_stats

    # snr
    if model.pspecific is not None:
        data.loc["SNR", "Mean"] = model.snr().mean().item()

    # classification statistics
    if model.pspecific is not None and model.data.ontarget.labels is not None:
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
        samples = torch.masked_select(model.pspecific.cpu(), mask)
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
        # check convergence status
        data.loc["trained", "Mean"] = False
        for line in open(model.run_path / "run.log"):
            if "model converged" in line:
                data.loc["trained", "Mean"] = True
        torch.save(ci_stats, path / f"{model.name}-params.tpqr")
        if save_matlab:
            from scipy.io import savemat

            for param, field in ci_stats.items():
                if param in (
                    "d/m_probs",
                    "d/z_probs",
                    "d/j_probs",
                    "p(specific)",
                    "z_map",
                ):
                    ci_stats[param] = field.numpy()
                    continue
                for stat, value in field.items():
                    ci_stats[param][stat] = value.numpy()
            savemat(path / "params.mat", ci_stats)
        data.to_csv(
            path / "statistics.csv",
        )
