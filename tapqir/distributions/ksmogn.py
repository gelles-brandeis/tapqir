# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
KSMOGN
^^^^^^
"""

import pykeops
import torch
from pykeops.torch import Genred
from pyro.distributions import TorchDistribution
from torch.distributions import Categorical, constraints
from torch.distributions.utils import lazy_property

from .util import gaussian_spots

pykeops.set_verbose(False)


class KSMOGN(TorchDistribution):
    r"""
    K-Spots Marginalized Offset Gamma Noise Image Distribution.

    .. math::
        \mu^S_{\mathsf{pixelX}(i), \mathsf{pixelY}(j)} =
        \dfrac{m \cdot h}{2 \pi w^2}
        \exp{\left( -\dfrac{(i-x-x^\mathsf{target})^2 + (j-y-y^\mathsf{target})^2}{2 w^2} \right)}

    .. math::
        \mu^I = b + \sum_{\mathsf{spot}} \mu^S

    .. math::
        p(D|\mu^I, g) = \sum_\delta p(\delta) p(D|\mu^I, g, \delta) = \sum_\delta \delta_\mathsf{weights}
        \cdot \mathrm{Gamma}(D - \delta_\mathsf{samples} | \mu^I, g)

    **Reference**:

    1. Ordabayev YA, Friedman LJ, Gelles J, Theobald DL.
       Bayesian machine learning analysis of single-molecule fluorescence colocalization images.
       bioRxiv. 2021 Oct. doi: `10.1101/2021.09.30.462536 <https://doi.org/10.1101/2021.09.30.462536>`_.

    :param height: Integrated spot intensity. Should be broadcastable
        to ``batch_shape + (K,)``.
    :param width: Spot width. Should be broadcastable
        to ``batch_shape + (K,)``.
    :param x: Spot center on x-axis. Should be broadcastable
        to ``batch_shape + (K,)``.
    :param y: Spot center on y-axis. Should be broadcastable
        to ``batch_shape + (K,)``.
    :param target_locs: Target location. Should have
        the rightmost size ``2`` correspondnig to locations on
        x- and y-axes, and be broadcastable to ``batch_shape + (2,)``.
    :param background: Background intensity. Should
        be broadcastable to ``batch_shape``.
    :param gain: Camera gain.
    :param offset_samples: Offset samples from the empirical distribution.
    :param offset_logits: Offset log weights corresponding to the offset samples.
    :param int P: Number of pixels along the axis.
    :param m: Spot presence indicator. Should be broadcastable
        to ``batch_shape + (K,)``.
    :param alpha: Signal cross-talk coefficient matrix. Should be broadcastable
        to ``(Q, C)``.
    :param bool use_pykeops: Use pykeops as backend to marginalize out offset.
    """

    arg_constraints = {}
    support = constraints.positive

    def __init__(
        self,
        height: torch.Tensor,
        width: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        target_locs: torch.Tensor,
        background: torch.Tensor,
        gain: torch.Tensor,
        offset_samples: torch.Tensor,
        offset_logits: torch.Tensor,
        P: int,
        m: torch.Tensor = None,
        alpha: torch.Tensor = None,
        use_pykeops: bool = True,
        validate_args=None,
    ):

        # shapes for cosmos and crosstalk models
        self.height = height  # (N, F, C, K) or (N, F, Q, K)
        self.width = width  # (N, F, C, K) or (N, F, Q, K)
        self.x = x
        self.y = y
        self.target_locs = target_locs  # (N, F, C, 2)
        self.m = m  # (N, F, C, K) or (N, F, Q, K)
        self.background = background[..., None, None]  # (N, F, C, P, P)
        if alpha is not None:
            C = alpha.shape[-1]
            self.gain = gain[..., None, None, None]  # (1, K, P, P)
            self.height = (
                self.height.unsqueeze(-2) * alpha[..., None]
            )  # (N, F, Q, C, K)
            self.width = self.width.unsqueeze(-2)  # (N, F, Q, 1, K)
            self.x = self.x.unsqueeze(-2)  # (N, F, Q, 1, K)
            self.y = self.y.unsqueeze(-2)  # (N, F, Q, 1, K)
            self.m = self.m.unsqueeze(-2)  # (N, F, Q, 1, K)
            self.target_locs = self.target_locs.unsqueeze(-3)  # (N, F, 1, C, 2)
        else:
            self.gain = gain[..., None, None]  # (1, P, P)
        self.alpha = alpha
        self.rate = 1 / self.gain

        self.offset_samples = offset_samples
        self.offset_logits = offset_logits
        self.P = P
        self.use_pykeops = use_pykeops
        if self.use_pykeops:
            device = self.target_locs.device.type
            self.device_pykeops = "GPU" if device == "cuda" else "CPU"

        # calculate batch shape
        batch_shape = torch.broadcast_shapes(
            height.shape, width.shape, x.shape, y.shape
        )  # (N, F, C, K) or (N, F, Q, K)
        if m is not None:
            batch_shape = torch.broadcast_shapes(
                batch_shape, m.shape
            )  # (N, F, C, K) or (N, F, Q, K)

        event_shape = torch.Size([P, P])  # (P, P)
        bg_shape = background.shape  # (N, F, C)
        target_shape = target_locs.shape[:-1]  # (N, F, C)
        # remove K dim
        batch_shape = batch_shape[:-1]  # (N, F, C) or (N, F, Q)
        if alpha is not None:
            # remove Q dim
            batch_shape = batch_shape[:-1]  # (N, F)
            # add C dim
            event_shape = (C,) + event_shape  # (C, P, P)
            # remove C dim
            bg_shape = bg_shape[:-1]  # (N, F)
            target_shape = target_shape[:-1]  # (N, F)
        batch_shape = torch.broadcast_shapes(
            batch_shape, bg_shape, target_shape
        )  # (N, F, C) or (N, F)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def gaussians(self):
        return gaussian_spots(
            self.height,  # (N, F, C, K) or (N, F, Q, 1, K)
            self.width,  # (N, F, C, K) or (N, F, Q, 1, K)
            self.x,  # (N, F, C, K) or (N, F, Q, 1, K)
            self.y,  # (N, F, C, K) or (N, F, Q, 1, K)
            self.target_locs.unsqueeze(-2),  # (N, F, C, 1, 2) or (N, F, 1, C, 1, 2)
            self.P,
            self.m,
        )

    @lazy_property
    def image(self):
        gaussians = self.gaussians  # (N, F, C, K, P, P) or (N, F, Q, C, K, P, P)
        if self.alpha is not None:
            # sum over the signal from multiple fluorescent dyes
            gaussians = gaussians.sum(-5)
        # sum the background and all the spots
        return self.background + gaussians.sum(-3)

    @lazy_property
    def concentration(self):
        return self.image / self.gain

    def rsample(self, sample_shape=torch.Size()):
        odx = (
            Categorical(logits=self.offset_logits)
            .expand(self.batch_shape + self.event_shape)
            .sample()
        )
        offset = self.offset_samples[odx]
        shape = self._extended_shape(sample_shape)
        value = torch._standard_gamma(
            self.concentration.expand(shape)
        ) / self.rate.expand(shape)
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value + offset

    def log_prob(self, value):
        if self.use_pykeops:
            formula = "wj+Log(Step(xi-gj-IntCst(1)))+(ai-IntCst(1))*Log(IfElse(xi-gj-IntCst(1),xi-gj,xi))-bi*(xi-gj)"
            variables = [
                "wj = Vj(1)",
                "gj = Vj(1)",
                "ai = Vi(1)",
                "bi = Vi(1)",
                "xi = Vi(1)",
            ]
            dtype = self.concentration.dtype
            my_routine = Genred(
                formula,
                variables,
                reduction_op="LogSumExp",
                axis=1,
            )
            concentration, value, rate = torch.broadcast_tensors(
                self.concentration, value, self.rate
            )
            shape = value.shape
            result = my_routine(
                self.offset_logits.reshape(-1, 1),
                self.offset_samples.reshape(-1, 1).to(dtype),
                concentration.reshape(-1, 1),
                rate.reshape(-1, 1).contiguous(),
                value.reshape(-1, 1).to(dtype),
                backend=self.device_pykeops,
            )
            result = result.reshape(shape)
            result = (
                self.concentration * torch.log(self.rate)
                - torch.lgamma(self.concentration)
                + result
            )
        else:
            value = torch.as_tensor(value).unsqueeze(-1)
            concentration = self.concentration.unsqueeze(-1)
            mask = value > self.offset_samples
            new_value = torch.where(
                mask, value - self.offset_samples, value.new_ones(())
            )
            obs_logits = (
                concentration * torch.log(self.rate)
                + (concentration - 1) * torch.log(new_value)
                - self.rate * (new_value)
                - torch.lgamma(concentration)
            )
            result = obs_logits + self.offset_logits + torch.log(mask)
            result = torch.logsumexp(result, -1)
        event_dims = tuple(-i for i in range(1, len(self.event_shape) + 1))
        return result.sum(event_dims)
