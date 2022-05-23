# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from pyro.ops.indexing import Vindex
from pyro.ops.stats import hpdi, quantile
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from tapqir.distributions.util import gaussian_spots
from tapqir.handlers import StatsMessenger

logger = logging.getLogger(__name__)


def snr_and_chi2(
    data: torch.Tensor,
    height: torch.Tensor,
    width: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target_locs: torch.Tensor,
    background: torch.Tensor,
    gain: float,
    offset_mean: float,
    offset_var: float,
    P: int,
    theta_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Calculate the signal-to-noise ratio.

    Total signal:

    .. math::
        \mu_{knf} =  \sum_{ij} I_{nfij}
        \mathcal{N}(i, j \mid x_{knf}, y_{knf}, w_{knf})`

    Noise:

    .. math::
        \sigma^2_{knf} = \sigma^2_{\text{offset}}
        + \mu_{knf} \text{gain}`

    Signal-to-noise ratio:

    .. math::
        \text{SNR}_{knf} =
        \dfrac{\mu_{knf} - b_{nf} - \mu_{\text{offset}}}{\sigma_{knf}}
        \text{ for } \theta_{nf} = k`
    """
    gaussians = gaussian_spots(
        height,
        width,
        x,
        y,
        target_locs,
        P,
    )

    # snr
    weights = gaussians / height[..., None, None]
    signal = ((data - background[..., None, None] - offset_mean) * weights).sum(
        dim=(-2, -1)
    )
    noise = (offset_var + background * gain).sqrt()
    snr_result = signal / noise

    # chi2 test
    img_ideal = background[..., None, None] + gaussians.sum(-4)
    chi2_result = (data - img_ideal - offset_mean) ** 2 / img_ideal

    return snr_result, chi2_result.mean(dim=(-1, -2))


def save_stats(model, path, CI=0.95, save_matlab=False):
    # global parameters
    global_params = model._global_params
    summary = pd.DataFrame(
        index=global_params,
        columns=["Mean", f"{int(100*CI)}% LL", f"{int(100*CI)}% UL"],
    )

    logger.info("- credible intervals")
    nbatch_size = model.nbatch_size
    fbatch_size = model.fbatch_size
    model.nbatch_size = None
    model.fbatch_size = None
    with StatsMessenger(
        CI=CI, K=model.K, N=model.data.Nt, F=model.data.F, Q=model.Q
    ) as ci_stats:
        model.guide()
    model.nbatch_size = nbatch_size
    model.fbatch_size = fbatch_size

    for param in global_params:
        if ci_stats[param]["Mean"].ndim == 0:
            summary.loc[param, "Mean"] = ci_stats[param]["Mean"].item()
            summary.loc[param, "95% LL"] = ci_stats[param]["LL"].item()
            summary.loc[param, "95% UL"] = ci_stats[param]["UL"].item()
        else:
            summary.loc[param, "Mean"] = ci_stats[param]["Mean"].tolist()
            summary.loc[param, "95% LL"] = ci_stats[param]["LL"].tolist()
            summary.loc[param, "95% UL"] = ci_stats[param]["UL"].tolist()
    logger.info("- spot probabilities")
    ci_stats["m_probs"] = model.m_probs.data.cpu()
    ci_stats["theta_probs"] = model.theta_probs.data.cpu()
    ci_stats["z_probs"] = model.z_probs.data.cpu()
    ci_stats["z_map"] = model.z_map.data.cpu()
    ci_stats["p_specific"] = ci_stats["theta_probs"].sum(0)

    # calculate vmin/vmax
    theta_mask = ci_stats["theta_probs"] > 0.5
    if theta_mask.sum():
        hmax = np.percentile(ci_stats["height"]["Mean"][theta_mask], 99)
    else:
        hmax = 1
    ci_stats["height"]["vmin"] = -0.03 * hmax
    ci_stats["height"]["vmax"] = 1.3 * hmax
    ci_stats["width"]["vmin"] = 0.5
    ci_stats["width"]["vmax"] = 2.5
    ci_stats["x"]["vmin"] = -9
    ci_stats["x"]["vmax"] = 9
    ci_stats["y"]["vmin"] = -9
    ci_stats["y"]["vmax"] = 9
    bmax = np.percentile(ci_stats["background"]["Mean"].flatten(), 99)
    ci_stats["background"]["vmin"] = -0.03 * bmax
    ci_stats["background"]["vmax"] = 1.3 * bmax

    # timestamps
    if model.data.time1 is not None:
        ci_stats["time1"] = model.data.time1
    if model.data.ttb is not None:
        ci_stats["ttb"] = model.data.ttb

    # intensity of target-specific spots
    theta_mask = torch.argmax(ci_stats["theta_probs"], dim=0)
    h_specific = Vindex(ci_stats["height"]["Mean"])[
        theta_mask, torch.arange(model.data.Nt)[:, None], torch.arange(model.data.F)
    ]
    ci_stats["h_specific"] = h_specific * (ci_stats["z_map"] > 0).long()

    model.params = ci_stats

    logger.info("- SNR and Chi2-test")
    # snr and chi2 test
    snr = torch.zeros(model.K, model.data.Nt, model.data.F, device=torch.device("cpu"))
    chi2 = torch.zeros(model.data.Nt, model.data.F, device=torch.device("cpu"))
    for n in range(model.data.Nt):
        snr[:, n], chi2[n] = snr_and_chi2(
            model.data.images[n, :, model.cdx],
            ci_stats["height"]["Mean"][:, n],
            ci_stats["width"]["Mean"][:, n],
            ci_stats["x"]["Mean"][:, n],
            ci_stats["y"]["Mean"][:, n],
            model.data.xy[n, :, model.cdx],
            ci_stats["background"]["Mean"][n],
            ci_stats["gain"]["Mean"],
            model.data.offset.mean,
            model.data.offset.var,
            model.data.P,
            ci_stats["theta_probs"][:, n],
        )
    snr_masked = snr[ci_stats["theta_probs"] > 0.5]
    summary.loc["SNR", "Mean"] = snr_masked.mean().item()
    ci_stats["chi2"] = {}
    ci_stats["chi2"]["values"] = chi2
    cmax = quantile(ci_stats["chi2"]["values"].flatten(), 0.99)
    ci_stats["chi2"]["vmin"] = -0.03 * cmax
    ci_stats["chi2"]["vmax"] = 1.3 * cmax

    # classification statistics
    if model.data.labels is not None:
        pred_labels = model.z_map[model.data.is_ontarget].cpu().numpy().ravel()
        true_labels = model.data.labels["z"][: model.data.N, :, model.cdx].ravel()

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

        mask = torch.from_numpy(model.data.labels["z"][: model.data.N, :, model.cdx])
        samples = torch.masked_select(model.z_probs[model.data.is_ontarget].cpu(), mask)
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
        param_path = path / f"{model.full_name}-params.tpqr"
        torch.save(ci_stats, param_path)
        logger.info(f"Parameters were saved in {param_path}")
        if save_matlab:
            from scipy.io import savemat

            for param, field in ci_stats.items():
                if param in (
                    "m_probs",
                    "theta_probs",
                    "z_probs",
                    "z_map",
                    "p_specific",
                    "h_specific",
                    "time1",
                    "ttb",
                ):
                    ci_stats[param] = np.asarray(field)
                    continue
                for stat, value in field.items():
                    ci_stats[param][stat] = np.asarray(value)
            mat_path = path / f"{model.full_name}-params.mat"
            savemat(mat_path, ci_stats)
            logger.info(f"Matlab parameters were saved in {mat_path}")
        csv_path = path / f"{model.full_name}-summary.csv"
        summary.to_csv(csv_path)
        logger.info(f"Summary statistics were saved in {csv_path}")
