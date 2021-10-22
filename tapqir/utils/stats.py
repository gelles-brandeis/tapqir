# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.ops.indexing import Vindex
from pyro.ops.stats import hpdi, quantile
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from tapqir.distributions import AffineBeta


def save_stats(model, path, CI=0.95, save_matlab=False):
    # global parameters
    global_params = model._global_params
    summary = pd.DataFrame(
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

    ci_stats = defaultdict(partial(defaultdict, list))
    num_samples = 10000
    for param in global_params:
        if param == "gain":
            fn = dist.Gamma(
                pyro.param("gain_loc") * pyro.param("gain_beta"),
                pyro.param("gain_beta"),
            )
        elif param == "pi":
            fn = dist.Dirichlet(pyro.param("pi_mean") * pyro.param("pi_size"))
        elif param == "lamda":
            fn = dist.Gamma(
                pyro.param("lamda_loc") * pyro.param("lamda_beta"),
                pyro.param("lamda_beta"),
            )
        elif param == "proximity":
            fn = AffineBeta(
                pyro.param("proximity_loc"),
                pyro.param("proximity_size"),
                0,
                (model.data.P + 1) / math.sqrt(12),
            )
        elif param == "trans":
            fn = dist.Dirichlet(
                pyro.param("trans_mean") * pyro.param("trans_size")
            ).to_event(1)
        else:
            raise NotImplementedError
        samples = fn.sample((num_samples,)).data.squeeze()
        ci_stats[param] = {}
        LL, UL = hpdi(
            samples,
            CI,
            dim=0,
        )
        ci_stats[param]["LL"] = LL.cpu()
        ci_stats[param]["UL"] = UL.cpu()
        ci_stats[param]["Mean"] = fn.mean.data.squeeze().cpu()

        # calculate Keq
        if param == "pi":
            ci_stats["Keq"] = {}
            ci_stats["Keq"]["LL"], ci_stats["Keq"]["UL"] = hpdi(
                samples[:, 1] / (1 - samples[:, 1]), CI, dim=0
            )
            ci_stats["Keq"]["Mean"] = (samples[:, 1] / (1 - samples[:, 1])).mean()

    # this does not need to be very accurate
    num_samples = 1000
    for param in local_params:
        LL, UL, Mean = [], [], []
        for ndx in torch.split(torch.arange(model.data.ontarget.N), model.nbatch_size):
            ndx = ndx[:, None]
            kdx = torch.arange(model.K)[:, None, None]
            ll, ul, mean = [], [], []
            for fdx in torch.split(
                torch.arange(model.data.ontarget.F), model.fbatch_size
            ):
                if param == "d/background":
                    fn = dist.Gamma(
                        Vindex(pyro.param("d/b_loc"))[ndx, fdx]
                        * Vindex(pyro.param("d/b_beta"))[ndx, fdx],
                        Vindex(pyro.param("d/b_beta"))[ndx, fdx],
                    )
                elif param == "d/height":
                    fn = dist.Gamma(
                        Vindex(pyro.param("d/h_loc"))[kdx, ndx, fdx]
                        * Vindex(pyro.param("d/h_beta"))[kdx, ndx, fdx],
                        Vindex(pyro.param("d/h_beta"))[kdx, ndx, fdx],
                    )
                elif param == "d/width":
                    fn = AffineBeta(
                        Vindex(pyro.param("d/w_mean"))[kdx, ndx, fdx],
                        Vindex(pyro.param("d/w_size"))[kdx, ndx, fdx],
                        0.75,
                        2.25,
                    )
                elif param == "d/x":
                    fn = AffineBeta(
                        Vindex(pyro.param("d/x_mean"))[kdx, ndx, fdx],
                        Vindex(pyro.param("d/size"))[kdx, ndx, fdx],
                        -(model.data.P + 1) / 2,
                        (model.data.P + 1) / 2,
                    )
                elif param == "d/y":
                    fn = AffineBeta(
                        Vindex(pyro.param("d/y_mean"))[kdx, ndx, fdx],
                        Vindex(pyro.param("d/size"))[kdx, ndx, fdx],
                        -(model.data.P + 1) / 2,
                        (model.data.P + 1) / 2,
                    )
                else:
                    raise NotImplementedError
                samples = fn.sample((num_samples,)).data.squeeze()
                l, u = hpdi(
                    samples,
                    CI,
                    dim=0,
                )
                m = fn.mean.data.squeeze()
                ll.append(l)
                ul.append(u)
                mean.append(m)
            else:
                LL.append(torch.cat(ll, -1))
                UL.append(torch.cat(ul, -1))
                Mean.append(torch.cat(mean, -1))
        else:
            ci_stats[param]["LL"] = torch.cat(LL, -2).cpu()
            ci_stats[param]["UL"] = torch.cat(UL, -2).cpu()
            ci_stats[param]["Mean"] = torch.cat(Mean, -2).cpu()

    for param in global_params:
        if param == "pi":
            summary.loc[param, "Mean"] = ci_stats[param]["Mean"][1].item()
            summary.loc[param, "95% LL"] = ci_stats[param]["LL"][1].item()
            summary.loc[param, "95% UL"] = ci_stats[param]["UL"][1].item()
            # Keq
            summary.loc["Keq", "Mean"] = ci_stats["Keq"]["Mean"].item()
            summary.loc["Keq", "95% LL"] = ci_stats["Keq"]["LL"].item()
            summary.loc["Keq", "95% UL"] = ci_stats["Keq"]["UL"].item()
        elif param == "trans":
            summary.loc["kon", "Mean"] = ci_stats[param]["Mean"][0, 1].item()
            summary.loc["kon", "95% LL"] = ci_stats[param]["LL"][0, 1].item()
            summary.loc["kon", "95% UL"] = ci_stats[param]["UL"][0, 1].item()
            summary.loc["koff", "Mean"] = ci_stats[param]["Mean"][1, 0].item()
            summary.loc["koff", "95% LL"] = ci_stats[param]["LL"][1, 0].item()
            summary.loc["koff", "95% UL"] = ci_stats[param]["UL"][1, 0].item()
        else:
            summary.loc[param, "Mean"] = ci_stats[param]["Mean"].item()
            summary.loc[param, "95% LL"] = ci_stats[param]["LL"].item()
            summary.loc[param, "95% UL"] = ci_stats[param]["UL"].item()
    if model.pspecific is not None:
        ci_stats["d/m_probs"] = model.m_probs.data.cpu()
        ci_stats["d/theta_probs"] = model.theta_probs.data.cpu()
        ci_stats["d/j_probs"] = model.j_probs.data.cpu()
        ci_stats["p(specific)"] = model.pspecific.data.cpu()
        ci_stats["z_map"] = model.z_map.data.cpu()

    model.params = ci_stats

    # snr
    #  if model.pspecific is not None:
    #      summary.loc["SNR", "Mean"] = model.snr().mean().item()

    # classification statistics
    if model.pspecific is not None and model.data.ontarget.labels is not None:
        pred_labels = model.z_map.cpu().numpy().ravel()
        true_labels = model.data.ontarget.labels["z"].ravel()

        with np.errstate(divide="ignore", invalid="ignore"):
            summary.loc["MCC", "Mean"] = matthews_corrcoef(true_labels, pred_labels)
        summary.loc["Recall", "Mean"] = recall_score(
            true_labels, pred_labels, zero_division=0
        )
        summary.loc["Precision", "Mean"] = precision_score(
            true_labels, pred_labels, zero_division=0
        )

        (
            summary.loc["TN", "Mean"],
            summary.loc["FP", "Mean"],
            summary.loc["FN", "Mean"],
            summary.loc["TP", "Mean"],
        ) = confusion_matrix(true_labels, pred_labels, labels=(0, 1)).ravel()

        mask = torch.from_numpy(model.data.ontarget.labels["z"][:, :, model.cdx])
        samples = torch.masked_select(model.pspecific.cpu(), mask)
        if len(samples):
            z_ll, z_ul = hpdi(samples, CI)
            summary.loc["p(specific)", "Mean"] = quantile(samples, 0.5).item()
            summary.loc["p(specific)", "95% LL"] = z_ll.item()
            summary.loc["p(specific)", "95% UL"] = z_ul.item()
        else:
            summary.loc["p(specific)", "Mean"] = 0.0
            summary.loc["p(specific)", "95% LL"] = 0.0
            summary.loc["p(specific)", "95% UL"] = 0.0

    model.summary = summary

    if path is not None:
        path = Path(path)
        # check convergence status
        #  data.loc["trained", "Mean"] = False
        #  for line in open(model.run_path / ".tapqir" / "log"):
        #      if "model converged" in line:
        #          data.loc["trained", "Mean"] = True
        torch.save(ci_stats, path / f"{model.full_name}-params.tpqr")
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
            savemat(path / f"{model.full_name}-params.mat", ci_stats)
        summary.to_csv(
            path / f"{model.full_name}-summary.csv",
        )
