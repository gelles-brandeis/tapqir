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


def truncated_poisson_probs(lamda, K, dtype):
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


def probs_m(lamda, S, K, dtype):
    r"""
    Spot presence probability :math:`p(m)`.

    .. math::

        p(m_{\mathsf{spot}(k)}) =
        \begin{cases}
            \mathbf{Bernoulli}(1) & \text{$\theta = k$} \\
            \mathbf{Bernoulli} \left( \sum_{l=1}^K
                \dfrac{l \cdot \mathbf{TruncPoisson}(l; \lambda, K)}{K} \right)
                & \text{$\theta = 0$} \rule{0pt}{4ex} \\
            \mathbf{Bernoulli} \left( \sum_{l=1}^{K-1}
                \dfrac{l \cdot \mathbf{TruncPoisson}(l; \lambda, K-1)}{K-1} \right)
                & \text{otherwise} \rule{0pt}{4ex}
        \end{cases}
    """
    result = torch.zeros(1 + K * S, K, dtype=dtype)
    kdx = torch.arange(K)
    tr_pois_km1 = truncated_poisson_probs(lamda, K - 1, dtype)
    km1 = torch.arange(1, K)
    result[:, :] = (km1 * tr_pois_km1[km1]).sum() / (K - 1)
    # theta == 0
    tr_pois_k = truncated_poisson_probs(lamda, K, dtype)
    k = torch.arange(1, K + 1)
    result[0] = (k * tr_pois_k[k]).sum() / K
    # theta == k
    result[kdx + 1, kdx] = 1
    return result


def probs_theta(pi, S, K, dtype):
    r"""
    Target-specific spot index probability :math:`p(\theta)`.

    .. math::

        p(\theta) = \mathbf{Categorical}\left(1 - \pi, \frac{\pi}{K}, \dots, \frac{\pi}{K}\right)
    """
    # 0 (False) - offtarget
    # 1 (True) - ontarget
    result = torch.zeros(2, K * S + 1, dtype=dtype)
    result[0, 0] = 1
    result[1, 0] = pi[0]
    for s in range(S):
        for k in range(K):
            result[1, K * s + k + 1] = pi[s + 1] / K
    return result
