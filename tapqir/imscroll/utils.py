# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

from pyroapi import pyro
from scipy.io import savemat


def save_matlab(model, path):
    path = Path(path)
    # save parameters in matlab format
    keys = ["h_loc", "w_mean", "x_mean", "y_mean", "b_loc"]
    matlab = {k: pyro.param(f"d/{k}").data.cpu().numpy() for k in keys}
    matlab[
        "parametersDescription"
    ] = "Parameters for N x F x K spots. \
        N - target sites, F - frames, K - max number of spots in the image. \
        h_loc - mean intensity, w_mean - mean spot width, \
        x_mean - x position, y_mean - y position, \
        b_loc - background intensity."
    matlab["z_probs"] = model.z_probs.cpu().numpy()
    matlab["j_probs"] = model.j_probs.cpu().numpy()
    matlab["m_probs"] = model.m_probs.cpu().numpy()
    matlab["z_marginal"] = model.z_marginal.cpu().numpy()
    matlab[
        "probabilitiesDescription"
    ] = "Probabilities for N x F x K spots. \
        z_probs - on-target spot probability, \
        j_probs - off-target spot probability, \
        m_probs - spot probability (on-target + off-target), \
        z_marginal - total on-target spot probability (sum of z_probs)."
    savemat(path / "parameters.mat", matlab)
