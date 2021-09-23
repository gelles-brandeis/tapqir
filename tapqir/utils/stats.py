# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.ops.stats import hpdi, quantile
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from tapqir.distributions import AffineBeta


def save_stats(model, path, CI=0.95, save_matlab=False):
    # change device to cpu
    model.to("cpu")
    # do calculations for all AOIs
    model.batch_size = model.n = None
    # global parameters
    global_params = model._global_params
    data = pd.DataFrame(
        index=global_params,
        columns=["Mean", f"{int(100*CI)}% LL", f"{int(100*CI)}% UL"],
    )
    # local parameters
    local_params = [
        "d/height",
        "d/width",
        "d/x",
        "d/y",
        "d/background",
    ]

    ci_stats = {}
    num_samples = 10000
    for param in global_params:
        if param == "gain":
            fn = dist.Gamma(
                pyro.param("gain_loc").cpu() * pyro.param("gain_beta").cpu(),
                pyro.param("gain_beta").cpu(),
            )
        elif param == "pi":
            fn = dist.Dirichlet(
                pyro.param("pi_mean").cpu() * pyro.param("pi_size").cpu()
            )
        elif param == "lamda":
            fn = dist.Gamma(
                pyro.param("lamda_loc").cpu() * pyro.param("lamda_beta").cpu(),
                pyro.param("lamda_beta").cpu(),
            )
        elif param == "proximity":
            fn = AffineBeta(
                pyro.param("proximity_loc").cpu(),
                pyro.param("proximity_size").cpu(),
                0,
                (model.data.P + 1) / math.sqrt(12),
            )
        else:
            raise NotImplementedError
        samples = fn.sample((num_samples,)).data.squeeze()
        ci_stats[param] = {}
        ci_stats[param]["LL"], ci_stats[param]["UL"] = hpdi(
            samples,
            CI,
            dim=0,
        )
        ci_stats[param]["Mean"] = fn.mean.data.squeeze()

        # calculate Keq
        if param == "pi":
            ci_stats["Keq"] = {}
            ci_stats["Keq"]["LL"], ci_stats["Keq"]["UL"] = hpdi(
                samples[:, 1] / (1 - samples[:, 1]), CI, dim=0
            )
            ci_stats["Keq"]["Mean"] = (samples[:, 1] / (1 - samples[:, 1])).mean()

    # this does not need to be very accurate
    num_samples = 500
    for param in local_params:
        if param == "d/background":
            fn = dist.Gamma(
                pyro.param("d/b_loc").cpu() * pyro.param("d/b_beta").cpu(),
                pyro.param("d/b_beta").cpu(),
            )
        elif param == "d/height":
            fn = dist.Gamma(
                pyro.param("d/h_loc").cpu() * pyro.param("d/h_beta").cpu(),
                pyro.param("d/h_beta").cpu(),
            )
        elif param == "d/width":
            fn = AffineBeta(
                pyro.param("d/w_mean").cpu(),
                pyro.param("d/w_size").cpu(),
                0.75,
                2.25,
            )
        elif param == "d/x":
            fn = AffineBeta(
                pyro.param("d/x_mean").cpu(),
                pyro.param("d/size").cpu(),
                -(model.data.P + 1) / 2,
                (model.data.P + 1) / 2,
            )
        elif param == "d/y":
            fn = AffineBeta(
                pyro.param("d/y_mean").cpu(),
                pyro.param("d/size").cpu(),
                -(model.data.P + 1) / 2,
                (model.data.P + 1) / 2,
            )
        else:
            raise NotImplementedError
        samples = fn.sample((num_samples,)).data.squeeze()
        ci_stats[param] = {}
        ci_stats[param]["LL"], ci_stats[param]["UL"] = hpdi(
            samples,
            CI,
            dim=0,
        )
        ci_stats[param]["Mean"] = fn.mean.data.squeeze()

    for param in global_params:
        if param == "pi":
            data.loc[param, "Mean"] = ci_stats[param]["Mean"][1].item()
            data.loc[param, "95% LL"] = ci_stats[param]["LL"][1].item()
            data.loc[param, "95% UL"] = ci_stats[param]["UL"][1].item()
            # Keq
            data.loc["Keq", "Mean"] = ci_stats["Keq"]["Mean"].item()
            data.loc["Keq", "95% LL"] = ci_stats["Keq"]["LL"].item()
            data.loc["Keq", "95% UL"] = ci_stats["Keq"]["UL"].item()
        else:
            data.loc[param, "Mean"] = ci_stats[param]["Mean"].item()
            data.loc[param, "95% LL"] = ci_stats[param]["LL"].item()
            data.loc[param, "95% UL"] = ci_stats[param]["UL"].item()
    if model.pspecific is not None:
        ci_stats["d/m_probs"] = model.m_probs.data.cpu()
        ci_stats["d/theta_probs"] = model.theta_probs.data.cpu()
        ci_stats["d/j_probs"] = model.j_probs.data.cpu()
        ci_stats["p(specific)"] = model.pspecific.data.cpu()
        ci_stats["z_map"] = model.z_map.data.cpu()

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
            z_ll, z_ul = hpdi(samples, CI)
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
        #  data.loc["trained", "Mean"] = False
        #  for line in open(model.run_path / ".tapqir" / "log"):
        #      if "model converged" in line:
        #          data.loc["trained", "Mean"] = True
        torch.save(ci_stats, path / f"{model.name}-params.tpqr")
        if save_matlab:
            from scipy.io import savemat

            for param, field in ci_stats.items():
                if param in (
                    "d/m_probs",
                    "d/theta_probs",
                    "d/j_probs",
                    "p(specific)",
                    "z_map",
                ):
                    ci_stats[param] = field.numpy()
                    continue
                for stat, value in field.items():
                    ci_stats[param][stat] = value.numpy()
            savemat(path / f"{model.name}-params.mat", ci_stats)
        data.to_csv(
            path / "statistics.csv",
        )
