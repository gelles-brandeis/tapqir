# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch


def _gaussian_spots(height, width, x, y, target_locs, P, m=None):
    r"""
    Calculates ideal shape of the 2D-Gaussian spots given spot parameters
    and target positions.

        :math:`\dfrac{h}{2 \pi \cdot w^2} \exp{\left( -\dfrac{(i-x)^2 + (j-y)^2}{2 \cdot w^2} \right)}`
    """
    # create meshgrid of PxP pixel positions
    P_range = torch.arange(P)
    i_pixel, j_pixel = torch.meshgrid(P_range, P_range, indexing="xy")
    ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)

    # Ideal 2D gaussian spots
    spot_locs = target_locs + torch.stack((x, y), -1)
    scale = width[..., None, None, None]
    loc = spot_locs[..., None, None, :]
    var = scale ** 2
    normalized_gaussian = torch.exp(
        (
            -((ij_pixel - loc) ** 2) / (2 * var)
            - scale.log()
            - math.log(math.sqrt(2 * math.pi))
        ).sum(-1)
    )
    if m is not None:
        height = m * height
    return height[..., None, None] * normalized_gaussian


def _truncated_poisson_probs(lamda, K, dtype):
    r"""
    Probability of the number of non-specific spots (TruncatedPoisson)
    for cases when :math:`\theta = 0` and :math:`\theta `.

    .. math::
        \mathbf{TruncPoisson}(\lambda, K)
        = 1 - e^{-\lambda} \sum_{i=0}^{K-1} \dfrac{\lambda^i}{i!} \mathrm{if } k = K
        \mathrm{; }
        \dfrac{\lambda^k e^{-\lambda}}{k!} \mathrm{otherwise}
    """
    result = torch.zeros(K + 1, dtype=dtype)
    kdx = torch.arange(K)
    result[:-1] = torch.exp(kdx.xlogy(lamda) - lamda - (kdx + 1).lgamma())
    result[-1] = 1 - result[:-1].sum()
    return result
