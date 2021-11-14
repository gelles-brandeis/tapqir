# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
Distribution utility functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import math

import torch


def gaussian_spots(
    height: torch.Tensor,
    width: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target_locs: torch.Tensor,
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


def probs_m(lamda: torch.Tensor, S: int, K: int) -> torch.Tensor:
    r"""
    Spot presence probability :math:`p(m | \theta, \lambda)`.

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
    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :return: A tensor of a shape ``lamda.shape + (1 + KS, K)`` of probabilities.
    """
    shape = lamda.shape + (1 + K * S, K)
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


def probs_theta(pi: torch.Tensor, S: int, K: int) -> torch.Tensor:
    r"""
    Target-specific spot index probability :math:`p(\theta)`.

    .. math::

        p(\theta) =
        \begin{cases}
            \mathbf{Categorical}\left(1 - \pi, \frac{\pi}{K}, \dots, \frac{\pi}{K}\right) & \textrm{if on-target} \\
            \mathbf{Categorical}\left(1, 0, \dots, 0\right) & \textrm{if off-target}
        \end{cases}

    :param pi: Average binding probability of target-specific binding.
    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :return: A tensor of a shape ``pi.shape[:-1] + (2, 1 + KS)`` of probabilities for off-target (``0``)
        and on-target (``1``) AOI.
    """
    shape = pi.shape[:-1] + (2, 1 + K * S)
    dtype = pi.dtype
    result = torch.zeros(shape, dtype=dtype)
    result[..., 0, 0] = 1
    result[..., 1, 0] = pi[..., 0]
    for s in range(S):
        for k in range(K):
            result[..., 1, K * s + k + 1] = pi[..., s + 1] / K
    return result


def init_theta(pi: torch.Tensor, S: int, K: int) -> torch.Tensor:
    r"""
    Target-specific spot index probability :math:`p(\theta)`.

    .. math::

        p(\theta) =
        \begin{cases}
            \mathbf{Categorical}\left(1 - \pi, \frac{\pi}{K}, \dots, \frac{\pi}{K}\right) & \textrm{if on-target} \\
            \mathbf{Categorical}\left(1, 0, \dots, 0\right) & \textrm{if off-target}
        \end{cases}

    :param pi: Initial probabilities.
    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :return: A tensor of a shape ``pi.shape[:-1] + (2, 1 + KS)`` of probabilities for off-target (``0``)
        and on-target (``1``) AOI.
    """
    shape = pi.shape[:-1] + (2, 1 + K * S)
    dtype = pi.dtype
    result = torch.zeros(shape, dtype=dtype)
    result[..., 0, 0] = 1
    result[..., 1, 0] = pi[..., 0]
    for s in range(S):
        for k in range(K):
            result[..., 1, K * s + k + 1] = pi[..., s + 1] / K
    return result


def trans_theta(A: torch.Tensor, S: int, K: int) -> torch.Tensor:
    r"""
    Target-specific spot index probability :math:`p(\theta)`.

    .. math::

        p(\theta) =
        \begin{cases}
            \mathbf{Categorical}\left(1 - \pi, \frac{\pi}{K}, \dots, \frac{\pi}{K}\right) & \textrm{if on-target} \\
            \mathbf{Categorical}\left(1, 0, \dots, 0\right) & \textrm{if off-target}
        \end{cases}

    :param A: Transition probabilities.
    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :return: A tensor of a shape ``A.shape[:-1] + (2, 1 + KS)`` of probabilities for off-target (``0``)
        and on-target (``1``) AOI.
    """
    shape = A.shape[:-2] + (2, 1 + K * S, 1 + K * S)
    dtype = A.dtype
    result = torch.zeros(shape, dtype=dtype)
    result[..., :, 0] = 1
    for i in range(1 + K * S):
        # FIXME
        j = (i + 1) // K
        result[..., 1, i, 0] = A[..., j, 0]
        for s in range(S):
            for k in range(K):
                result[..., 1, i, K * s + k + 1] = A[..., j, s + 1] / K
    return result
