# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math
from functools import reduce

import torch
from pykeops.torch import Genred
import pykeops
pykeops.clean_pykeops()
from pyro.distributions import Gamma, TorchDistribution
from torch.distributions import constraints


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


class KSpotGamma(TorchDistribution):
    r"""
    Offset + Background + K-number of Gaussian Spots Image Model with
    Gamma distributed noise.

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
        gain,
        P=None,
        m=None,
        offset_samples=None,
        offset_logits=None,
        validate_args=None,
    ):

        #  if P is None:
        #      P = offset.shape[-1]
        gaussian_spots = _gaussian_spots(
            height, width, x, y, target_locs.unsqueeze(-2), P, m
        )
        image = background[..., None, None] + gaussian_spots.sum(-3)

        self.offset_samples = offset_samples
        self.offset_logits = offset_logits
        self.concentration = image / gain
        self.rate = 1 / gain
        batch_shape = image.shape[:-2]
        event_shape = image.shape[-2:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

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
        # value = value.reshape(-1, 1)
        # formula = "m-Square(x-loc)/(IntCst(2)*var)-Log(Sqrt(var))-p"
        formula = "wj+(ai-IntCst(1))*Log(xi-gj)-b*(xi-gj)"
        variables = [
            "wj = Vj(1)",
            "gj = Vj(1)",
            "ai = Vi(1)",
            "b = Pm(1)",
            "xi = Vi(1)",
        ]

        my_routine = Genred(
            formula,
            variables,
            reduction_op="LogSumExp",
            axis=1,
            dtype="float64",
        )
        concentration, value = torch.broadcast_tensors(self.concentration, value)
        shape = value.shape
        result = my_routine(
            self.offset_logits.reshape(-1, 1),
            self.offset_samples.reshape(-1, 1).to(torch.float64),
            concentration.reshape(-1, 1),
            self.rate.reshape(1),
            value.reshape(-1, 1).to(torch.float64),
            backend="GPU",
        )
        result = result.masked_fill(torch.isnan(result), -40.)
        breakpoint()
        result = result.reshape(shape)
        result = (
            self.concentration * torch.log(self.rate)
            - torch.lgamma(self.concentration)
            + result
        )
        #  # sequential
        #  torch.cuda.empty_cache()
        #  self.dist = Gamma(self.concentration, self.rate, validate_args=False)
        #  #  mask = value > self.offset_samples
        #  #  value = torch.where(mask, value - self.offset_samples, value.new_ones(()))
        #  result = (
        #      logit + self.dist.log_prob(value - offset)
        #      for offset, logit in zip(self.offset_samples, self.offset_logits)
        #  )
        #  result = reduce(torch.logaddexp, result)
        # result = torch.logsumexp(result, -1)

        #  # vectorized
        #  value = torch.as_tensor(value).unsqueeze(-1)
        #  mask = value > self.offset_samples
        #  value = torch.where(mask, value - self.offset_samples, value.new_ones(()))
        #  concentration = self.concentration.unsqueeze(-1)
        #  obs_logits = (
        #      concentration * torch.log(self.rate)
        #      + (concentration - 1) * torch.log(value)
        #      - self.rate * (value)
        #      - torch.lgamma(concentration)
        #  )
        #  result = obs_logits + self.offset_logits
        #  result = result.masked_fill(~mask, -40.0)
        #  result = torch.logsumexp(result, -1)

        return result.sum((-2, -1))
