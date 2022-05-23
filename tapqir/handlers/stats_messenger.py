# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import re

import scipy.stats as stats
import torch
from pyro.poutine.messenger import Messenger
from pyroapi import distributions as dist


class StatsMessenger(Messenger):
    """
    Confidence interval with equal areas around the median.
    """

    def __init__(
        self,
        CI: float = 0.95,
        K: int = None,
        N: int = None,
        F: int = None,
        Q: int = None,
    ):
        super().__init__()
        self.CI = CI
        self.ci_stats = {}
        self.K = K
        self.N = N
        self.F = F
        self.Q = Q

    def _pyro_sample(self, msg):
        if (
            type(msg["fn"]).__name__ == "_Subsample"
            or msg["infer"].get("enumerate", None) == "parallel"
        ):
            return
        name = msg["name"]
        scipy_dist = torch_to_scipy_dist(msg["fn"])
        if scipy_dist is None:
            return
        LL, UL = scipy_dist.interval(alpha=self.CI)
        args = re.split("_k|_q", name)
        if len(args) == 1:
            (base_name,) = args
            self.ci_stats[base_name] = {}
            self.ci_stats[base_name]["LL"] = torch.as_tensor(
                LL, device=torch.device("cpu")
            )
            self.ci_stats[base_name]["UL"] = torch.as_tensor(
                UL, device=torch.device("cpu")
            )
            self.ci_stats[base_name]["Mean"] = msg["fn"].mean.detach().cpu()
        elif len(args) == 2:
            base_name, k = args
            k = int(k)
            assert self.Q == 1
            if k == 0:
                self.ci_stats[base_name] = {}
                self.ci_stats[base_name]["LL"] = torch.zeros(
                    self.K, self.N, self.F, device=torch.device("cpu")
                )
                self.ci_stats[base_name]["UL"] = torch.zeros(
                    self.K, self.N, self.F, device=torch.device("cpu")
                )
                self.ci_stats[base_name]["Mean"] = torch.zeros(
                    self.K, self.N, self.F, device=torch.device("cpu")
                )
            self.ci_stats[base_name]["LL"][k] = torch.as_tensor(
                LL, device=torch.device("cpu")
            )
            self.ci_stats[base_name]["UL"][k] = torch.as_tensor(
                UL, device=torch.device("cpu")
            )
            self.ci_stats[base_name]["Mean"][k] = msg["fn"].mean.detach().cpu()
        elif len(args) == 3:
            base_name, k, q = args
            k, q = int(k), int(q)
            assert self.Q > 1
            if (k == 0) and (q == 0):
                self.ci_stats[base_name] = {}
                self.ci_stats[base_name]["LL"] = torch.zeros(
                    self.K, self.N, self.F, self.Q, device=torch.device("cpu")
                )
                self.ci_stats[base_name]["UL"] = torch.zeros(
                    self.K, self.N, self.F, self.Q, device=torch.device("cpu")
                )
                self.ci_stats[base_name]["Mean"] = torch.zeros(
                    self.K, self.N, self.F, self.Q, device=torch.device("cpu")
                )
            self.ci_stats[base_name]["LL"][k, :, :, q] = torch.as_tensor(
                LL, device=torch.device("cpu")
            )
            self.ci_stats[base_name]["UL"][k, :, :, q] = torch.as_tensor(
                UL, device=torch.device("cpu")
            )
            self.ci_stats[base_name]["Mean"][k, :, :, q] = msg["fn"].mean.detach().cpu()
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
    elif isinstance(torch_dist, dist.Independent):
        return torch_to_scipy_dist(torch_dist.base_dist)
    elif isinstance(torch_dist, dist.Delta):
        return None
    else:
        raise NotImplementedError
