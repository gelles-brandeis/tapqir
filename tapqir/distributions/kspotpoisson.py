# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import torch
import torch.nn.functional as F
from pyro.distributions import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import lazy_property, probs_to_logits


def _gaussian_spots(height, width, x, y, target_locs, P, m=None):
    r"""
    Calculates ideal shape of the 2D-Gaussian spots given spot parameters
    and target positions.

        :math:`\dfrac{h}{2 \pi \cdot w^2} \exp{\left( -\dfrac{(i-x)^2 + (j-y)^2}{2 \cdot w^2} \right)}`
    """
    # create meshgrid of PxP pixel positions
    P_range = torch.arange(P)
    i_pixel, j_pixel = torch.meshgrid(P_range, P_range)
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


class KSpotNB(TorchDistribution):
    r"""
    Offset + Background + K-number of Gaussian Spots Image Model with
    NegativeBinomial distributed noise.

    :param torch.Tensor height: Integrated spot intensity. Should be broadcastable
        to ``(batch_shape, K)``.
    :param torch.Tensor width: Spot width. Should be broadcastable
        to ``(batch_shape, K)``.
    :param torch.Tensor x: Spot center on x-axis. Should be broadcastable
        to ``(batch_shape, K)``.
    :param torch.Tensor y: Spot center on y-axis. Should be broadcastable
        to ``(batch_shape, K)``.
    :param torch.Tensor target_locs: Target location. Should have
        the rightmost size ``2`` correspondnig to locations on
        x- and y-axes, and be broadcastable to ``(batch_shape,)``.
    :param torch.Tensor background: Background intensity. Should
        be broadcastable to ``(batch_shape,)``.
    :param torch.Tensor offset: Offset intensity. Should be broadcastable
        to ``(batch_shape, P, P)``.
    :param torch.Tensor m: Spot presence indicator. Should be broadcastable
        to ``(batch_shape, K)``.
    :param torch.Tensor gain: Camera gain.
    :param int P: Number of pixels along the axis.
    """

    arg_constraints = {}
    support = constraints.positive

    def __init__(
        self,
        height,
        width,
        x,
        y,
        target_locs,
        background,
        offset,
        gain,
        P=None,
        m=None,
        validate_args=None,
    ):

        if P is None:
            P = offset.shape[-1]
        gaussian_spots = _gaussian_spots(
            height, width, x, y, target_locs.unsqueeze(-2), P, m
        )
        image = background[..., None, None] + gaussian_spots.sum(-3)

        self.offset = offset
        self.total_count = image / (gain - 1)
        self.probs = (gain - 1) / gain
        self.concentration = image / gain
        self.rate = 1 / gain
        batch_shape = image.shape[:-2]
        event_shape = image.shape[-2:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        value = torch._standard_gamma(
            self.concentration.expand(shape)
        ) / self.rate.expand(shape)
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value + self.offset

    def log_prob(self, value):
        mask = value >= self.offset
        value = torch.where(mask, value - self.offset, value.new_zeros(()))
        log_unnormalized_prob = self.total_count * F.logsigmoid(
            -self.logits
        ) + value * F.logsigmoid(self.logits)

        log_normalization = (
            -torch.lgamma(self.total_count + value)
            + torch.lgamma(1.0 + value)
            + torch.lgamma(self.total_count)
        )

        result = log_unnormalized_prob - log_normalization
        result = torch.where(mask, result, result.new_zeros(()))

        #  result2 = (
        #      self.concentration * torch.log(self.rate)
        #      + (self.concentration - 1) * torch.log(value)
        #      - self.rate * (value)
        #      - torch.lgamma(self.concentration)
        #  )
        #  result2 = torch.where(mask, result2, result.new_zeros(()))
        #  breakpoint()
        #  result2 = result2.sum((-2, -1))
        return result.sum((-2, -1))
