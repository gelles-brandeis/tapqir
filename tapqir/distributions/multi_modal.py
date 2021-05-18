# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
from pyro.distributions import Gamma, TorchDistribution


class MultiModal(TorchDistribution):
    r"""
    Sum of the offset variable and the Gamma distributed variable::
    """

    def __init__(self, loc, beta, weights):
        self.weights = weights
        self.dist = Gamma(loc * beta, beta)
        batch_shape = torch.Size([])
        event_shape = torch.Size([])
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        value = value.unsqueeze(-1)

        obs_logits = self.dist.log_prob(value)
        result = obs_logits + self.logits
        result = torch.logsumexp(result, -1)
        return result
