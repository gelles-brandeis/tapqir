# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import torch
import torch.distributions.constraints as constraints
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property

from tapqir.distributions import AffineBeta, KSpotGammaNoise
from tapqir.models.model import Model


class Cosmos(Model):
    """
    Time-independent Single Molecule Colocalization Model.
    """

    name = "cosmos"

    def __init__(self, S=1, K=2, device="cpu", dtype="double", marginal=False):
        super().__init__(S, K, device, dtype)
        if marginal:
            self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
            self._global_params = ["gain", "proximity", "lamda", "pi"]
            self._classify = False
        else:
            self.conv_params = ["-ELBO"]
            self._classify = True

    def TraceELBO(self, jit=False):
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=2, ignore_jit_warnings=True
        )

    @property
    def probs_j(self):
        result = torch.zeros(2, self.K + 1, dtype=self.dtype)
        result[0, : self.K] = torch.exp(
            self.lamda.log() * torch.arange(self.K)
            - self.lamda
            - torch.arange(1, self.K + 1).lgamma()
        )
        result[0, -1] = 1 - result[0, : self.K].sum()
        result[1, : self.K - 1] = torch.exp(
            self.lamda.log() * torch.arange(self.K - 1)
            - self.lamda
            - torch.arange(1, self.K).lgamma()
        )
        result[1, -2] = 1 - result[0, : self.K - 1].sum()
        return result

    @property
    def probs_m(self):
        # this only works for K=2
        result = torch.zeros(1 + self.K * self.S, self.K, 2, dtype=self.dtype)
        probs_j = self.probs_j
        result[0, :, 0] = probs_j[0, 0] + probs_j[0, 1] / 2
        result[0, :, 1] = probs_j[0, 2] + probs_j[0, 1] / 2
        result[1, 0, 1] = 1
        result[1, 1, 0] = probs_j[1, 0]
        result[1, 1, 1] = probs_j[1, 1]
        result[2, 0, 0] = probs_j[1, 0]
        result[2, 0, 1] = probs_j[1, 1]
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

    @property
    def z_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        return pyro.param("d/theta_probs").data[..., 1:].permute(2, 0, 1)

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
        return torch.einsum(
            "sknf,nfs->knf",
            pyro.param("d/m_probs").data[..., 1],
            pyro.param("d/theta_probs").data,
        )

    @property
    def pspecific(self):
        return self.z_probs.sum(-3)

    @property
    def z_map(self):
        return self.pspecific > 0.5

    def model(self):
        # global parameters
        self.gain = pyro.sample("gain", dist.HalfNormal(50)).squeeze()
        self.state_model()

        # test data
        self.spot_model(self.data.ontarget, prefix="d")

        # control data
        if self.data.offtarget.images is not None:
            self.spot_model(self.data.offtarget, prefix="c")

    def state_model(self):
        self.pi = pyro.sample(
            "pi", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1))
        ).squeeze()
        self.lamda = pyro.sample("lamda", dist.Exponential(1)).squeeze()
        self.proximity = pyro.sample("proximity", dist.Exponential(1)).squeeze()
        self.size = torch.stack(
            (
                torch.tensor(2.0),
                (((self.data.P + 1) / (2 * self.proximity)) ** 2 - 1),
            ),
            dim=-1,
        )

    @property
    def infer_config(self):
        if self._classify:
            return {
                "expose_types": ["sample"],
                "expose": ["d/theta_probs", "d/m_probs"],
            }
        return {"expose_types": ["sample", "param"]}

    def guide(self):
        with handlers.block(**self.infer_config):
            # global parameters
            pyro.sample(
                "gain",
                dist.Gamma(
                    pyro.param("gain_loc").to(self.device)
                    * pyro.param("gain_beta").to(self.device),
                    pyro.param("gain_beta").to(self.device),
                ),
            )
            self.state_guide()

            # test data
            self.spot_guide(self.data.ontarget, prefix="d")

            # control data
            if self.data.offtarget.images is not None:
                self.spot_guide(self.data.offtarget, prefix="c")

    def state_guide(self):
        pyro.sample(
            "pi",
            dist.Dirichlet(
                pyro.param("pi_mean").to(self.device)
                * pyro.param("pi_size").to(self.device)
            ),
        )
        pyro.sample(
            "lamda",
            dist.Gamma(
                pyro.param("lamda_loc").to(self.device)
                * pyro.param("lamda_beta").to(self.device),
                pyro.param("lamda_beta").to(self.device),
            ),
        )
        pyro.sample(
            "proximity",
            AffineBeta(
                pyro.param("proximity_loc").to(self.device),
                pyro.param("proximity_size").to(self.device),
                0,
                (self.data.P + 1) / math.sqrt(12),
            ),
        )

    def spot_model(self, data, prefix):
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
        frames = pyro.plate(f"{prefix}/frames", data.F, subsample=self.f, dim=-1)

        with aois as ndx:
            # background mean and std
            background_mean = pyro.sample(
                f"{prefix}/background_mean", dist.HalfNormal(1000)
            )
            background_std = pyro.sample(
                f"{prefix}/background_std", dist.HalfNormal(100)
            )
            with frames as fdx:
                # sample background intensity
                background = pyro.sample(
                    f"{prefix}/background",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
                    ),
                )

                # sample hidden model state (1+K*S,)
                if prefix == "d":
                    if self._classify:
                        theta = pyro.sample(
                            f"{prefix}/theta",
                            dist.Categorical(self.probs_theta),
                        )
                    else:
                        theta = pyro.sample(
                            f"{prefix}/theta",
                            dist.Categorical(self.probs_theta),
                            infer={"enumerate": "parallel"},
                        )
                else:
                    theta = 0

                ms, heights, widths, xs, ys = [], [], [], [], []
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

                    # append
                    ms.append(m)
                    heights.append(height)
                    widths.append(width)
                    xs.append(x)
                    ys.append(y)

                # subtract offset
                odx = pyro.sample(
                    f"{prefix}/offset",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )
                offset = self.data.offset.samples[odx]
                # fetch data
                obs, target_locs = data.fetch(ndx[:, None], fdx)
                # observed data
                pyro.sample(
                    f"{prefix}/data",
                    KSpotGammaNoise(
                        torch.stack(heights, -1),
                        torch.stack(widths, -1),
                        torch.stack(xs, -1),
                        torch.stack(ys, -1),
                        target_locs,
                        background,
                        offset,
                        self.gain,
                        data.P,
                        torch.stack(torch.broadcast_tensors(*ms), -1),
                    ),
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
        frames = pyro.plate(f"{prefix}/frames", data.F, subsample=self.f, dim=-1)

        with aois as ndx:
            pyro.sample(
                f"{prefix}/background_mean",
                dist.Delta(
                    pyro.param(f"{prefix}/background_mean_loc")[ndx].to(self.device)
                ),
            )
            pyro.sample(
                f"{prefix}/background_std",
                dist.Delta(
                    pyro.param(f"{prefix}/background_std_loc")[ndx].to(self.device)
                ),
            )
            with frames as fdx:
                # sample background intensity
                pyro.sample(
                    f"{prefix}/background",
                    dist.Gamma(
                        Vindex(pyro.param(f"{prefix}/b_loc"))[ndx[:, None], fdx].to(
                            self.device
                        )
                        * Vindex(pyro.param(f"{prefix}/b_beta"))[ndx[:, None], fdx].to(
                            self.device
                        ),
                        Vindex(pyro.param(f"{prefix}/b_beta"))[ndx[:, None], fdx].to(
                            self.device
                        ),
                    ),
                )
                if self._classify and prefix == "d":
                    theta = pyro.sample(
                        f"{prefix}/theta",
                        dist.Categorical(
                            Vindex(pyro.param(f"{prefix}/theta_probs"))[
                                ndx[:, None], fdx
                            ].to(self.device)
                        ),
                        infer={"enumerate": "parallel"},
                    )

                for kdx in spots:
                    # sample spot presence m
                    if prefix == "d":
                        if self._classify:
                            m_probs = Vindex(pyro.param(f"{prefix}/m_probs"))[
                                theta, kdx, ndx[:, None], fdx
                            ].to(self.device)
                        else:
                            m_probs = torch.einsum(
                                "snft,nfs->nft",
                                Vindex(pyro.param(f"{prefix}/m_probs"))[
                                    torch.arange(self.S * self.K + 1)[:, None, None],
                                    kdx,
                                    ndx[:, None],
                                    fdx,
                                    :,
                                ].to(self.device),
                                Vindex(pyro.param(f"{prefix}/theta_probs"))[
                                    ndx[:, None], fdx, :
                                ].to(self.device),
                            )
                    else:
                        m_probs = Vindex(pyro.param(f"{prefix}/m_probs"))[
                            kdx, ndx[:, None], fdx
                        ].to(self.device)
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
                                Vindex(pyro.param(f"{prefix}/h_loc"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device)
                                * Vindex(pyro.param(f"{prefix}/h_beta"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                                Vindex(pyro.param(f"{prefix}/h_beta"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/width_{kdx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/w_mean"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                                Vindex(pyro.param(f"{prefix}/w_size"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                                0.75,
                                2.25,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/x_{kdx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/x_mean"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                                Vindex(pyro.param(f"{prefix}/size"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/y_{kdx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/y_mean"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                                Vindex(pyro.param(f"{prefix}/size"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
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

    def init_parameters(self):
        device = self.device
        pyro.param(
            "proximity_loc",
            lambda: torch.tensor(0.5, device=device),
            constraint=constraints.interval(
                0,
                (self.data.P + 1) / math.sqrt(12) - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "proximity_size",
            lambda: torch.tensor(100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "lamda_loc",
            lambda: torch.tensor(0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "lamda_beta",
            lambda: torch.tensor(100, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "pi_mean",
            lambda: torch.ones(self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "pi_size",
            lambda: torch.tensor(2, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "gain_loc",
            lambda: torch.tensor(5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "gain_beta",
            lambda: torch.tensor(100, device=device),
            constraint=constraints.positive,
        )

        self.spot_parameters(self.data.ontarget, prefix="d")

        if self.data.offtarget.images is not None:
            self.spot_parameters(self.data.offtarget, prefix="c")

    def spot_parameters(self, data, prefix):
        device = self.device
        pyro.param(
            f"{prefix}/background_mean_loc",
            lambda: torch.full(
                (data.N, 1),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/background_std_loc",
            lambda: torch.ones(data.N, 1, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            f"{prefix}/b_loc",
            lambda: torch.full(
                (data.N, data.F),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/b_beta",
            lambda: torch.ones(data.N, data.F, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_loc",
            lambda: torch.full((self.K, data.N, data.F), 2000, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_beta",
            lambda: torch.full((self.K, data.N, data.F), 0.001, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/w_mean",
            lambda: torch.full((self.K, data.N, data.F), 1.5, device=device),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/w_size",
            lambda: torch.full((self.K, data.N, data.F), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            f"{prefix}/x_mean",
            lambda: torch.zeros(self.K, data.N, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/y_mean",
            lambda: torch.zeros(self.K, data.N, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/size",
            lambda: torch.full((self.K, data.N, data.F), 200, device=device),
            constraint=constraints.greater_than(2.0),
        )

        # classification
        if prefix == "d":
            pyro.param(
                "d/theta_probs",
                lambda: torch.ones(data.N, data.F, 1 + self.K * self.S, device=device),
                constraint=constraints.simplex,
            )
            m_probs = torch.ones(
                1 + self.K * self.S,
                self.K,
                data.N,
                data.F,
                2,
                device=device,
            )
            m_probs[1, 0, :, :, 0] = 0
            m_probs[2, 1, :, :, 0] = 0
            pyro.param("d/m_probs", lambda: m_probs, constraint=constraints.simplex)
        else:
            pyro.param(
                "c/m_probs",
                lambda: torch.ones(self.K, data.N, data.F, 2, device=device),
                constraint=constraints.simplex,
            )
