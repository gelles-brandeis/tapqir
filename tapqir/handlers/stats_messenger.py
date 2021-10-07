# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from pyro.ops.stats import hpdi
from pyro.poutine.messenger import Messenger


class StatsMessenger(Messenger):
    def __init__(self, sites, CI=0.95, num_samples=500):
        super().__init__()
        self.sites = sites
        self.CI = CI
        self.num_samples = num_samples
        self.stats = {}

    def _pyro_sample(self, msg):
        if (
            type(msg["fn"]).__name__ == "_Subsample"
            or msg["infer"].get("enumerate", None) == "parallel"
        ):
            return
        name = msg["name"]
        if name in self.sites:
            self.stats[name] = {}
            samples = msg["fn"].sample((self.num_samples,)).data.squeeze().cpu()
            self.stats[name]["LL"], self.stats[name]["UL"] = hpdi(
                samples,
                self.CI,
                dim=0,
            )
            self.stats[name]["Mean"] = msg["fn"].mean.data.squeeze().cpu()

            # calculate Keq
            if name == "pi":
                self.stats["Keq"] = {}
                self.stats["Keq"]["LL"], self.stats["Keq"]["UL"] = hpdi(
                    samples[:, 1] / (1 - samples[:, 1]), self.CI, dim=0
                )
                self.stats["Keq"]["Mean"] = (samples[:, 1] / (1 - samples[:, 1])).mean()
        msg["stop"] = True
        msg["done"] = True

    def __enter__(self):
        super().__enter__()
        return self.stats
