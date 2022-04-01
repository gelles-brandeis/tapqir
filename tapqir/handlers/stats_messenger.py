# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import scipy.stats as stats
import torch
from pyro.poutine.messenger import Messenger
from pyroapi import distributions as dist


class StatsMessenger(Messenger):
    """
    Confidence interval with equal areas around the median.
    """

    def __init__(self, CI=0.95):
        super().__init__()
        self.CI = CI
        self.ci_stats = {}

    def _pyro_sample(self, msg):
        if (
            type(msg["fn"]).__name__ == "_Subsample"
            or msg["infer"].get("enumerate", None) == "parallel"
            or isinstance(msg["fn"], dist.Delta)
        ):
            return
        name = msg["name"]
        self.ci_stats[name] = {}
        scipy_dist = torch_to_scipy_dist(msg["fn"])
        LL, UL = scipy_dist.interval(alpha=self.CI)
        self.ci_stats[name]["LL"] = torch.as_tensor(LL, device=torch.device("cpu"))
        self.ci_stats[name]["UL"] = torch.as_tensor(UL, device=torch.device("cpu"))
        self.ci_stats[name]["Mean"] = msg["fn"].mean.detach().cpu()
        msg["stop"] = True
        msg["done"] = True

    def __enter__(self):
        super().__enter__()
        return self.ci_stats


def torch_to_scipy_dist(torch_dist):
    if isinstance(torch_dist, dist.Gamma):
        return stats.gamma(
            torch_dist.concentration.detach().cpu(),
            scale=1 / torch_dist.rate.detach().cpu(),
        )
    elif isinstance(torch_dist, dist.Beta):
        return stats.beta(
            a=torch_dist.concentration1.detach().cpu(),
            b=torch_dist.concentration0.detach().cpu(),
        )
    elif isinstance(torch_dist, dist.AffineBeta):
        return stats.beta(
            a=torch_dist.concentration1.detach().cpu(),
            b=torch_dist.concentration0.detach().cpu(),
            loc=torch_dist.loc.detach().cpu(),
            scale=torch_dist.scale.detach().cpu(),
        )
    elif isinstance(torch_dist, dist.Dirichlet):
        return stats.beta(
            a=torch_dist.concentration.detach().cpu(),
            b=(
                torch_dist.concentration.sum(-1, keepdim=True).detach()
                - torch_dist.concentration.detach()
            ).cpu(),
        )
    else:
        raise NotImplementedError
