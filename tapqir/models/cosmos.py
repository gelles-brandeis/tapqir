# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import torch
import torch.distributions.constraints as constraints
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property
from tqdm import tqdm

from tapqir.distributions import AffineBeta
from tapqir.models.model import Model


class Cosmos(Model):
    r"""
    for :math:`n=1` to :math:`N`:

        for :math:`n=1` to :math:`F`:

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`
    """
    name = "cosmos"

    def __init__(self, S=1, K=2, device="cpu", dtype="double"):
        super().__init__(S, K, device, dtype)

    @lazy_property
    def num_states(self):
        r"""
        Total number of states for the image model given by:

            :math:`2 (1+SK) K`
        """
        return 2 * (1 + self.K * self.S) * self.K

    @property
    def probs_j(self):
        result = torch.zeros(2, self.K + 1, dtype=self.dtype)
        result[0, : self.K] = (
            dist.Poisson(self.lamda).log_prob(torch.arange(self.K).to(self.dtype)).exp()
        )
        result[0, -1] = 1 - result[0, : self.K].sum()
        result[1, : self.K - 1] = (
            dist.Poisson(self.lamda)
            .log_prob(torch.arange(self.K - 1).to(self.dtype))
            .exp()
        )
        result[1, -2] = 1 - result[0, : self.K - 1].sum()
        return result

    @property
    def probs_m(self):
        # this only works for K=2
        result = torch.zeros(1 + self.K * self.S, self.K, 2, dtype=self.dtype)
        result[0, :, 0] = self.probs_j[0, 0] + self.probs_j[0, 1] / 2
        result[0, :, 1] = self.probs_j[0, 2] + self.probs_j[0, 1] / 2
        result[1, 0, 1] = 1
        result[1, 1, 0] = self.probs_j[1, 0]
        result[1, 1, 1] = self.probs_j[1, 1]
        result[2, 0, 0] = self.probs_j[1, 0]
        result[2, 0, 1] = self.probs_j[1, 1]
        result[2, 1, 1] = 1
        return result

    @property
    def probs_theta(self):
        result = torch.zeros(self.K * self.S + 1, dtype=self.dtype)
        result[0] = self.pi[0]
        for s in range(self.S):
            for k in range(self.K):
                result[self.K * s + k + 1] = self.pi[s + 1] / self.K
        return result

    @lazy_property
    def theta_to_z(self):
        result = torch.zeros(self.K * self.S + 1, self.K, dtype=torch.long)
        for s in range(self.S):
            result[1 + s * self.K : 1 + (s + 1) * self.K] = torch.eye(self.K) * (s + 1)
        return result

    @lazy_property
    def ontarget(self):
        return torch.clamp(self.theta_to_z, min=0, max=1)

    def compute_theta_samples(self, num_samples):
        samples = torch.zeros(
            num_samples, self.data.ontarget.N, self.data.ontarget.F
        ).long()
        batch_size = self.batch_size
        split_size = 1 + self.batch_size // 2
        self.batch_size = None
        for i in tqdm(range(num_samples)):
            for ndx in torch.split(torch.arange(self.data.ontarget.N), split_size):
                self.n = ndx
                guide_trace = handlers.trace(self.guide).get_trace()
                trained_model = handlers.replay(self.model, trace=guide_trace)
                inferred_model = infer.infer_discrete(
                    trained_model, temperature=1, first_available_dim=-3
                )
                trace = handlers.trace(inferred_model).get_trace()
                samples[i, ndx] = trace.nodes["d/theta"]["value"]
        self.theta_samples = samples
        self.batch_size = batch_size
        torch.save(self.theta_samples, self.path / "theta_samples.tpqr")

    @property
    def z_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        try:
            return (
                Vindex(self.theta_to_z)[self.theta_samples]
                .to(self.dtype)
                .mean(0)
                .permute(2, 0, 1)
            )
        except AttributeError:
            return torch.zeros(self.K, self.data.ontarget.N, self.data.ontarget.F)

    @property
    def j_probs(self):
        r"""
        Probability of an off-target spot :math:`p(j_{knf})`.
        """
        return self.m_probs - self.z_probs

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return pyro.param("d/m_probs").data[..., 1].to(self.device)

    @property
    def z_marginal(self):
        return self.z_probs.sum(-3)

    @property
    def z_map(self):
        return self.z_marginal > 0.5

    def model(self):
        # global parameters
        self.gain = pyro.sample("gain", dist.HalfNormal(50)).squeeze()
        self.lamda = pyro.sample("lamda", dist.Exponential(1)).squeeze()
        self.proximity = pyro.sample("proximity", dist.Exponential(1)).squeeze()
        self.size = torch.stack(
            (
                torch.tensor(2.0),
                (((self.data.P + 1) / (2 * self.proximity)) ** 2 - 1),
            ),
            dim=-1,
        )
        self.state_model()

        # test data
        self.spot_model(self.data.ontarget, prefix="d")

        # control data
        if self.data.offtarget.data is not None:
            self.spot_model(self.data.offtarget, prefix="c")

    def state_model(self):
        self.pi = pyro.sample(
            "pi", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1))
        ).squeeze()

    def guide(self):
        # initialize guide parameters
        self.guide_parameters()

        # global parameters
        pyro.sample(
            "gain",
            dist.Gamma(
                pyro.param("gain_loc") * pyro.param("gain_beta"),
                pyro.param("gain_beta"),
            ),
        )
        pyro.sample(
            "lamda",
            dist.Gamma(
                pyro.param("lamda_loc") * pyro.param("lamda_beta"),
                pyro.param("lamda_beta"),
            ),
        )
        pyro.sample(
            "proximity",
            AffineBeta(
                pyro.param("proximity_loc"),
                pyro.param("proximity_size"),
                0,
                (self.data.P + 1) / math.sqrt(12),
            ),
        )
        self.state_guide()

        # test data
        self.spot_guide(self.data.ontarget, prefix="d")

        # control data
        if self.data.offtarget.data is not None:
            self.spot_guide(self.data.offtarget, prefix="c")

    def state_guide(self):
        pyro.sample("pi", dist.Dirichlet(pyro.param("pi_mean") * pyro.param("pi_size")))

    def spot_model(self, data, prefix):
        # spots
        spots = pyro.plate(f"{prefix}/spots", self.K)
        # aoi sites
        aois = pyro.plate(f"{prefix}/aois", data.N, dim=-2)
        # time frames
        frames = pyro.plate(f"{prefix}/frames", data.F, dim=-1)

        with aois as ndx:
            # fetch data
            obs, target_locs = data[ndx]
            # background mean and std
            background_mean = pyro.sample(
                f"{prefix}/background_mean", dist.HalfNormal(1000)
            )
            background_std = pyro.sample(
                f"{prefix}/background_std", dist.HalfNormal(100)
            )
            with frames:
                # sample background intensity
                background = pyro.sample(
                    f"{prefix}/background",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
                    ),
                )
                locs = background[..., None, None]

                # sample hidden model state (1+K*S,)
                if prefix == "d":
                    theta = pyro.sample(
                        f"{prefix}/theta",
                        dist.Categorical(self.probs_theta),
                        infer={"enumerate": "parallel"},
                    )
                else:
                    theta = 0

                for kdx in spots:
                    ontarget = Vindex(self.ontarget)[theta, kdx]
                    # spot presence
                    m = pyro.sample(
                        f"{prefix}/m_{kdx}",
                        dist.Categorical(Vindex(self.probs_m)[theta, kdx]),
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        height = pyro.sample(
                            f"{prefix}/height_{kdx}",
                            dist.HalfNormal(10000),
                        )
                        width = pyro.sample(
                            f"{prefix}/width_{kdx}",
                            AffineBeta(
                                1.5,
                                2,
                                0.75,
                                2.25,
                            ),
                        )
                        x = pyro.sample(
                            f"{prefix}/x_{kdx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        y = pyro.sample(
                            f"{prefix}/y_{kdx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )

                    # calculate image shape w/o offset
                    height = height.masked_fill(m == 0, 0.0)
                    gaussian = self.gaussian(height, width, x, y, target_locs)
                    locs = locs + gaussian

                # subtract offset
                odx = pyro.sample(
                    f"{prefix}/offset",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )
                offset = self.data.offset.samples[odx]
                offset_mask = obs > offset
                obs = torch.where(offset_mask, obs - offset, obs.new_ones(()))
                # observed data
                pyro.sample(
                    f"{prefix}/data",
                    dist.Gamma(
                        locs / self.gain,
                        1 / self.gain,
                    )
                    .mask(mask=offset_mask)
                    .to_event(2),
                    obs=obs,
                )

    def spot_guide(self, data, prefix):
        # spots
        spots = pyro.plate(f"{prefix}/spots", self.K)
        # aoi sites
        aois = pyro.plate(
            f"{prefix}/aois",
            data.N,
            subsample_size=self.batch_size,
            subsample=self.n,
            dim=-2,
        )
        # time frames
        frames = pyro.plate(f"{prefix}/frames", data.F, dim=-1)

        with aois as ndx:
            pyro.sample(
                f"{prefix}/background_mean",
                dist.Delta(pyro.param(f"{prefix}/background_mean_loc")[ndx]),
            )
            pyro.sample(
                f"{prefix}/background_std",
                dist.Delta(pyro.param(f"{prefix}/background_std_loc")[ndx]),
            )
            with frames as fdx:
                # sample background intensity
                pyro.sample(
                    f"{prefix}/background",
                    dist.Gamma(
                        pyro.param(f"{prefix}/b_loc")[ndx]
                        * pyro.param(f"{prefix}/b_beta")[ndx],
                        pyro.param(f"{prefix}/b_beta")[ndx],
                    ),
                )

                for kdx in spots:
                    # sample spot presence m
                    m_probs = Vindex(pyro.param(f"{prefix}/m_probs"))[
                        kdx, ndx[:, None], fdx
                    ]
                    m = pyro.sample(
                        f"{prefix}/m_{kdx}",
                        dist.Categorical(m_probs),
                        infer={"enumerate": "parallel"},
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        pyro.sample(
                            f"{prefix}/height_{kdx}",
                            dist.Gamma(
                                pyro.param(f"{prefix}/h_loc")[kdx, ndx]
                                * pyro.param(f"{prefix}/h_beta")[kdx, ndx],
                                pyro.param(f"{prefix}/h_beta")[kdx, ndx],
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/width_{kdx}",
                            AffineBeta(
                                pyro.param(f"{prefix}/w_mean")[kdx, ndx],
                                pyro.param(f"{prefix}/w_size")[kdx, ndx],
                                0.75,
                                2.25,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/x_{kdx}",
                            AffineBeta(
                                pyro.param(f"{prefix}/x_mean")[kdx, ndx],
                                pyro.param(f"{prefix}/size")[kdx, ndx],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/y_{kdx}",
                            AffineBeta(
                                pyro.param(f"{prefix}/y_mean")[kdx, ndx],
                                pyro.param(f"{prefix}/size")[kdx, ndx],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                pyro.sample(
                    f"{prefix}/offset",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )

    def guide_parameters(self):
        pyro.param(
            "proximity_loc",
            lambda: torch.tensor(0.5),
            constraint=constraints.interval(
                0,
                (self.data.P + 1) / math.sqrt(12) - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "proximity_size",
            lambda: torch.tensor(100),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param("gain_loc", lambda: torch.tensor(5), constraint=constraints.positive)
        pyro.param(
            "gain_beta", lambda: torch.tensor(100), constraint=constraints.positive
        )
        pyro.param(
            "lamda_loc", lambda: torch.tensor(0.5), constraint=constraints.positive
        )
        pyro.param(
            "lamda_beta", lambda: torch.tensor(100), constraint=constraints.positive
        )

        self.state_parameters()

        self.spot_parameters(self.data.ontarget, prefix="d")

        if self.data.offtarget.data is not None:
            self.spot_parameters(self.data.offtarget, prefix="c")

    def state_parameters(self):
        pyro.param(
            "pi_mean", lambda: torch.ones(self.S + 1), constraint=constraints.simplex
        )
        pyro.param("pi_size", lambda: torch.tensor(2), constraint=constraints.positive)

    def spot_parameters(self, data, prefix):
        pyro.param(
            f"{prefix}/background_mean_loc",
            lambda: torch.full((data.N, 1), data.median - self.data.offset.median),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/background_std_loc",
            lambda: torch.ones(data.N, 1),
            constraint=constraints.positive,
        )

        pyro.param(
            f"{prefix}/m_probs",
            lambda: torch.ones(self.K, data.N, data.F, 2),
            constraint=constraints.simplex,
        )
        pyro.param(
            f"{prefix}/b_loc",
            lambda: torch.full((data.N, data.F), data.median - self.data.offset.median),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/b_beta",
            lambda: torch.ones(data.N, data.F),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_loc",
            lambda: torch.full((self.K, data.N, data.F), 2000.0),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_beta",
            lambda: torch.full((self.K, data.N, data.F), 0.001),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/w_mean",
            lambda: torch.full((self.K, data.N, data.F), 1.5),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/w_size",
            lambda: torch.full((self.K, data.N, data.F), 100.0),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            f"{prefix}/x_mean",
            lambda: torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/y_mean",
            lambda: torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        size = torch.ones(self.K, data.N, data.F) * 200.0
        if self.K == 2:
            size[1] = 7.0
        elif self.K == 3:
            size[1] = 7.0
            size[2] = 3.0
        pyro.param(
            f"{prefix}/size", lambda: size, constraint=constraints.greater_than(2.0)
        )
