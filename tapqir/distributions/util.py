# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
Distribution utility functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import math
from functools import lru_cache

import torch


def gaussian_spots(
    height: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    width: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    x: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    y: torch.Tensor,  # (N, F, C, K) or (N, F, Q, 1, K)
    target_locs: torch.Tensor,  # (N, F, C, 1, 2) or (N, F, 1, C, 1, 2)
    P: int,
    m: torch.Tensor = None,
) -> torch.Tensor:
    r"""
    Calculates ideal shape of the 2D-Gaussian spots given spot parameters
    and target positions.

    .. math::
        \mu^S_{\mathsf{pixelX}(i), \mathsf{pixelY}(j)} =
        \dfrac{m \cdot h}{2 \pi w^2}
        \exp{\left( -\dfrac{(i-x-x^\mathsf{target})^2 + (j-y-y^\mathsf{target})^2}{2 w^2} \right)}

    :param height: Integrated spot intensity. Should be broadcastable to ``batch_shape``.
    :param width: Spot width. Should be broadcastable to ``batch_shape``.
    :param x: Spot center on x-axis. Should be broadcastable to ``batch_shape``.
    :param y: Spot center on y-axis. Should be broadcastable to ``batch_shape``.
    :param target_locs: Target location. Should have
        the rightmost size ``2`` correspondnig to locations on
        x- and y-axes, and be broadcastable to ``batch_shape + (2,)``.
    :param P: Number of pixels along the axis.
    :param m: Spot presence indicator. Should be broadcastable to ``batch_shape``.
    :return: A tensor of a shape ``batch_shape + (P, P)`` representing 2D-Gaussian spots.
    """
    # create meshgrid of PxP pixel positions
    device = height.device
    P_range = torch.arange(P, device=device)
    i_pixel, j_pixel = torch.meshgrid(P_range, P_range, indexing="xy")
    ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)

    # Ideal 2D gaussian spots
    spot_locs = target_locs + torch.stack((x, y), -1)
    scale = width[..., None, None, None]
    loc = spot_locs[..., None, None, :]
    var = scale**2
    normalized_gaussian = torch.exp(
        (
            -((ij_pixel - loc) ** 2) / (2 * var)
            - scale.log()
            - math.log(math.sqrt(2 * math.pi))
        ).sum(-1)
    )  # (N, F, C, K, P, P) or (N, F, Q, C, K, P, P)
    if m is not None:
        height = m * height
    return height[..., None, None] * normalized_gaussian


def truncated_poisson_probs(lamda: torch.Tensor, K: int) -> torch.Tensor:
    r"""
    Probability of the number of non-specific spots.

    .. math::

        \mathbf{TruncatedPoisson}(\lambda, K) =
        \begin{cases}
            1 - e^{-\lambda} \sum_{i=0}^{K-1} \dfrac{\lambda^i}{i!} & \textrm{if } k = K \\
            \dfrac{\lambda^k e^{-\lambda}}{k!} & \mathrm{otherwise}
        \end{cases}

    :param lamda: Average rate of target-nonspecific binding.
    :param K: Maximum number of spots that can be present in a single image.
    :return: A tensor of a shape ``lamda.shape + (K+1,)`` of probabilities.
    """
    shape = lamda.shape + (K + 1,)
    dtype = lamda.dtype
    result = torch.zeros(shape, dtype=dtype)
    kdx = torch.arange(K)
    result[..., :-1] = torch.exp(
        kdx.xlogy(lamda.unsqueeze(-1)) - lamda.unsqueeze(-1) - (kdx + 1).lgamma()
    )
    result[..., -1] = 1 - result[..., :-1].sum(-1)
    return result


def probs_m(lamda: torch.Tensor, K: int) -> torch.Tensor:
    r"""
    Prior spot presence probability :math:`p(m | \theta, \lambda)`.

    .. math::

        p(m_{\mathsf{spot}(k)} | \theta, \lambda) =
        \begin{cases}
            \mathbf{Bernoulli}(1) & \text{$\theta = k$} \\
            \mathbf{Bernoulli} \left( \sum_{l=1}^K
                \dfrac{l \cdot \mathbf{TruncPoisson}(l; \lambda, K)}{K} \right)
                & \text{$\theta = 0$} \rule{0pt}{4ex} \\
            \mathbf{Bernoulli} \left( \sum_{l=1}^{K-1}
                \dfrac{l \cdot \mathbf{TruncPoisson}(l; \lambda, K-1)}{K-1} \right)
                & \text{otherwise} \rule{0pt}{4ex}
        \end{cases}

    :param lamda: Average rate of target-nonspecific binding.
    :param K: Maximum number of spots that can be present in a single image.
    :return: A tensor of a shape ``lamda.shape + (1 + K, K)`` of probabilities.
    """
    shape = lamda.shape + (1 + K, K)
    dtype = lamda.dtype
    result = torch.zeros(shape, dtype=dtype)
    kdx = torch.arange(K)
    tr_pois_km1 = truncated_poisson_probs(lamda, K - 1)
    km1 = torch.arange(1, K)
    result[..., :, :] = (km1 * tr_pois_km1[..., km1]).sum(-1).unsqueeze(-1).unsqueeze(
        -1
    ) / (K - 1)
    # theta == 0
    tr_pois_k = truncated_poisson_probs(lamda, K)
    k = torch.arange(1, K + 1)
    result[..., 0, :] = (k * tr_pois_k[..., k]).sum(-1).unsqueeze(-1) / K
    # theta == k
    result[..., kdx + 1, kdx] = 1
    return result


def expand_offtarget(probs: torch.Tensor) -> torch.Tensor:
    r"""
    Expand state probability ``probs`` (e.g., :math:`\pi` or :math:`A`) to off-target AOIs.

    .. math::

        p(\mathsf{state}) =
        \begin{cases}
            \mathbf{Categorical}\left( \mathsf{probs} \right) & \textrm{if on-target} \\
            \mathbf{Categorical}\left( \left[ 1, 0, \dots, 0 \right] \right) & \textrm{if off-target}
        \end{cases}

    :param probs: Probability of target-specific states.
    :return: A tensor of a shape ``probs.shape + (2,)`` of probabilities for off-target (``0``)
        and on-target (``1``) AOI.
    """
    offtarget_probs = torch.zeros_like(probs)
    offtarget_probs[..., 0] = 1
    return torch.stack([offtarget_probs, probs], dim=-1)


@lru_cache(maxsize=None)
def probs_theta(K: int, device: torch.device) -> torch.Tensor:
    r"""
    Prior probability for target-specific spot index :math:`p(\theta | z)`.

    .. math::
        p(\theta | z) =
        \begin{cases}
            \mathbf{Categorical}\left( \begin{bmatrix} 0 & 1/K & \dots & 1/K \end{bmatrix} \right) & z > 0 \\
            \mathbf{Categorical}\left( \begin{bmatrix} 1 & 0 & \dots & 0 \end{bmatrix} \right) & z = 0
        \end{cases}

    :param K: Maximum number of spots that can be present in a single image.
    :return: A tensor of a shape ``(2, 1 + K)`` of :math:`\theta` probabilities for spot-absent (``0``)
        and spot-present (``1``) cases.
    """
    result = torch.zeros(2, 1 + K, device=device)
    result[0, 0] = 1
    result[1, 1:] = 1 / K
    return result
