# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.distributions.constraints as constraints
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property

from tapqir.distributions import AffineBeta, CrossTalk, KSpotGammaNoise
from tapqir.models.model import Model


class MultiColor(Model):
    """
    Time-independent Single Molecule Colocalization Model.
    """

    # name = "multicolor"
    name = "crosstalk"

    def __init__(self, S=1, K=2, device="cpu", dtype="double", marginal=False):
        super().__init__(S, K, device, dtype)
        self._global_params = ["gain", "proximity", "lamda", "pi"]
        if marginal:
            self.conv_params = [
                "-ELBO",
                "proximity_loc_0",
                "proximity_loc_1",
                "gain_loc",
                "lamda_loc_0",
                "lamda_loc_1",
            ]
            self._classify = False
        else:
            self.conv_params = [
                "-ELBO",
                "proximity_loc_0",
                "proximity_loc_1",
                "gain_loc",
                "lamda_loc_0",
                "lamda_loc_1",
            ]
            self._classify = True

    def TraceELBO(self, jit=False):
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=2, ignore_jit_warnings=True
        )

    @property
    def probs_j(self):
        result = torch.zeros(2, self.K + 1, self.data.C, dtype=self.dtype)
        result[0, : self.K] = torch.exp(
            self.lamda.log() * torch.arange(self.K).unsqueeze(-1)
            - self.lamda
            - torch.arange(1, self.K + 1).lgamma().unsqueeze(-1)
        )
        result[0, -1] = 1 - result[0, : self.K].sum(0)
        result[1, : self.K - 1] = torch.exp(
            self.lamda.log() * torch.arange(self.K - 1).unsqueeze(-1)
            - self.lamda
            - torch.arange(1, self.K).lgamma().unsqueeze(-1)
        )
        result[1, -2] = 1 - result[0, : self.K - 1].sum(0)
        return result

    @property
    def probs_m(self):
        # this only works for K=2
        result = torch.zeros(
            1 + self.K * self.S, self.K, self.data.C, 2, dtype=self.dtype
        )
        probs_j = self.probs_j
        result[0, :, :, 0] = probs_j[0, 0] + probs_j[0, 1] / 2
        result[0, :, :, 1] = probs_j[0, 2] + probs_j[0, 1] / 2
        result[1, 0, :, 1] = 1
        result[1, 1, :, 0] = probs_j[1, 0]
        result[1, 1, :, 1] = probs_j[1, 1]
        result[2, 0, :, 0] = probs_j[1, 0]
        result[2, 0, :, 1] = probs_j[1, 1]
        result[2, 1, :, 1] = 1
        return result

    @property
    def probs_theta(self):
        result = torch.zeros(1 + self.S, self.K * self.S + 1, dtype=self.dtype)
        result[0, 0] = 1
        result[1, 1:] = 0.5
        return result

    @lazy_property
    def theta_onehot(self):
        result = torch.zeros(self.K * self.S + 1, self.K, dtype=torch.long)
        for s in range(self.S):
            result[1 + s * self.K : 1 + (s + 1) * self.K] = torch.eye(self.K) * (s + 1)
        return result

    @lazy_property
    def ontarget(self):
        return torch.clamp(self.theta_onehot, min=0, max=1)

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "sknf,nfs->knf",
            pyro.param("d/m_probs").data[..., 1],
            pyro.param("d/z_probs").data,
        )

    @property
    def pspecific(self):
        return pyro.param("d/z_probs")[..., 1].data

    @property
    def z_map(self):
        return self.pspecific > 0.5

    def model(self):
        # global parameters
        self.gain = pyro.sample("gain", dist.HalfNormal(50)).squeeze()
        self.crosstalk = pyro.sample(
            "crosstalk_sample", dist.HalfNormal(torch.tensor([0.5, 0.5])).to_event(1)
        )
        self.state_model()

        # test data
        self.spot_model(self.data.ontarget, prefix="d")

        # control data
        if self.data.offtarget.images is not None:
            self.spot_model(self.data.offtarget, prefix="c")

    def state_model(self):
        channels = pyro.plate("channels", 2, dim=-1)
        with channels:
            self.pi = pyro.sample(
                "pi", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1))
            ).squeeze()
            self.lamda = pyro.sample("lamda", dist.Exponential(1)).squeeze()
            self.proximity = pyro.sample("proximity", dist.Exponential(1)).squeeze()
        self.size = torch.stack(
            (
                torch.tensor([2.0, 2.0]),
                (((self.data.P + 1) / (2 * self.proximity)) ** 2 - 1),
            ),
            dim=-2,
        )

    @property
    def infer_config(self):
        if self._classify:
            return {
                "expose_types": ["sample"],
                "expose": ["d/z_probs", "d/m_probs"],
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
            pyro.sample(
                "crosstalk_sample", dist.Delta(pyro.param("crosstalk")).to_event(1)
            )
            self.state_guide()

            # test data
            self.spot_guide(self.data.ontarget, prefix="d")

            # control data
            if self.data.offtarget.images is not None:
                self.spot_guide(self.data.offtarget, prefix="c")

    def state_guide(self):
        channels = pyro.plate("channels", self.data.C, dim=-1)
        with channels:
            pyro.sample(
                "pi",
                dist.Dirichlet(
                    pyro.param("pi_mean").to(self.device)
                    * pyro.param("pi_size").to(self.device).unsqueeze(-1)
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

        ms, heights, widths, xs, ys, offsets, backgrounds = [], [], [], [], [], [], []
        for cdx in range(data.C):
            with aois as ndx:
                # background mean and std
                background_mean = pyro.sample(
                    f"{prefix}/background_mean_{cdx}", dist.HalfNormal(1000)
                )
                background_std = pyro.sample(
                    f"{prefix}/background_std_{cdx}", dist.HalfNormal(100)
                )
                with frames as fdx:
                    # sample background intensity
                    background = pyro.sample(
                        f"{prefix}/background_{cdx}",
                        dist.Gamma(
                            (background_mean / background_std) ** 2,
                            background_mean / background_std ** 2,
                        ),
                    )
                    backgrounds.append(background)

                    # sample hidden model state (1+K*S,)
                    if prefix == "d":
                        if self._classify:
                            z = pyro.sample(
                                f"{prefix}/z_{cdx}",
                                dist.Categorical(self.pi[cdx]),
                            )
                        else:
                            z = pyro.sample(
                                f"{prefix}/z_{cdx}",
                                dist.Categorical(self.pi[cdx]),
                                infer={"enumerate": "parallel"},
                            )
                        theta = pyro.sample(
                            f"{prefix}/theta_{cdx}",
                            dist.Categorical(self.probs_theta[z]),
                            infer={"enumerate": "parallel"},
                        )
                    else:
                        z = 0
                        theta = 0

                    for kdx in spots:
                        ontarget = Vindex(self.ontarget)[theta, kdx]
                        # spot presence
                        m = pyro.sample(
                            f"{prefix}/m_{kdx}_{cdx}",
                            dist.Categorical(Vindex(self.probs_m)[theta, kdx, cdx]),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            height = pyro.sample(
                                f"{prefix}/height_{kdx}_{cdx}",
                                dist.HalfNormal(10000),
                            )
                            width = pyro.sample(
                                f"{prefix}/width_{kdx}_{cdx}",
                                AffineBeta(
                                    1.5,
                                    2,
                                    0.75,
                                    2.25,
                                ),
                            )
                            x = pyro.sample(
                                f"{prefix}/x_{kdx}_{cdx}",
                                AffineBeta(
                                    0,
                                    Vindex(self.size)[ontarget, cdx],
                                    -(data.P + 1) / 2,
                                    (data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"{prefix}/y_{kdx}_{cdx}",
                                AffineBeta(
                                    0,
                                    Vindex(self.size)[ontarget, cdx],
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
                        f"{prefix}/offset_{cdx}",
                        dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                        .expand([data.P, data.P])
                        .to_event(2),
                    )
                    offset = self.data.offset.samples[odx]
                    offsets.append(offset)

        with aois as ndx, frames as fdx:
            # fetch data
            obs, target_locs = data.fetch(ndx[:, None], fdx)
            heights = torch.stack(
                (torch.stack(heights[:2], -1), torch.stack(heights[2:], -1)), -2
            )
            widths = torch.stack(
                (torch.stack(widths[:2], -1), torch.stack(widths[2:], -1)), -2
            )
            xs = torch.stack((torch.stack(xs[:2], -1), torch.stack(xs[2:], -1)), -2)
            ys = torch.stack((torch.stack(ys[:2], -1), torch.stack(ys[2:], -1)), -2)
            ms = torch.broadcast_tensors(*ms)
            ms = torch.stack((torch.stack(ms[:2], -1), torch.stack(ms[2:], -1)), -2)
            backgrounds = torch.stack(backgrounds, -1)
            offset = torch.stack(offsets, -3)
            # observed data
            pyro.sample(
                f"{prefix}/data",
                CrossTalk(
                    heights,
                    widths,
                    xs,
                    ys,
                    target_locs,
                    backgrounds,
                    offset,
                    self.gain,
                    self.crosstalk,
                    data.P,
                    ms,
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

        for cdx in range(data.C):
            with aois as ndx:
                ndx = ndx[:, None]
                pyro.sample(
                    f"{prefix}/background_mean_{cdx}",
                    dist.Delta(
                        Vindex(pyro.param(f"{prefix}/background_mean_loc"))[
                            ndx, 0, cdx
                        ].to(self.device)
                    ),
                )
                pyro.sample(
                    f"{prefix}/background_std_{cdx}",
                    dist.Delta(
                        Vindex(pyro.param(f"{prefix}/background_std_loc"))[
                            ndx, 0, cdx
                        ].to(self.device)
                    ),
                )
                with frames as fdx:
                    # sample background intensity
                    pyro.sample(
                        f"{prefix}/background_{cdx}",
                        dist.Gamma(
                            Vindex(pyro.param(f"{prefix}/b_loc"))[ndx, fdx, cdx].to(
                                self.device
                            )
                            * Vindex(pyro.param(f"{prefix}/b_beta"))[ndx, fdx, cdx].to(
                                self.device
                            ),
                            Vindex(pyro.param(f"{prefix}/b_beta"))[ndx, fdx, cdx].to(
                                self.device
                            ),
                        ),
                    )
                    if self._classify and prefix == "d":
                        z = pyro.sample(
                            f"{prefix}/z_{cdx}",
                            dist.Categorical(
                                Vindex(pyro.param(f"{prefix}/z_probs"))[
                                    ndx, fdx, cdx
                                ].to(self.device)
                            ),
                            infer={"enumerate": "parallel"},
                        )

                    for kdx in spots:
                        # sample spot presence m
                        if prefix == "d":
                            if self._classify:
                                m_probs = Vindex(pyro.param(f"{prefix}/m_probs"))[
                                    z,
                                    kdx,
                                    ndx,
                                    fdx,
                                    cdx,
                                ].to(self.device)
                            else:
                                m_probs = torch.einsum(
                                    "snft,nfs->nft",
                                    Vindex(pyro.param(f"{prefix}/m_probs"))[
                                        torch.arange(self.S + 1)[:, None, None],
                                        kdx,
                                        ndx,
                                        fdx,
                                        cdx,
                                    ].to(self.device),
                                    Vindex(pyro.param(f"{prefix}/z_probs"))[
                                        ndx, fdx, cdx
                                    ].to(self.device),
                                )
                        else:
                            m_probs = Vindex(pyro.param(f"{prefix}/m_probs"))[
                                kdx, ndx, fdx, cdx
                            ].to(self.device)
                        m = pyro.sample(
                            f"{prefix}/m_{kdx}_{cdx}",
                            dist.Categorical(m_probs),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"{prefix}/height_{kdx}_{cdx}",
                                dist.Gamma(
                                    Vindex(pyro.param(f"{prefix}/h_loc"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device)
                                    * Vindex(pyro.param(f"{prefix}/h_beta"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device),
                                    Vindex(pyro.param(f"{prefix}/h_beta"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device),
                                ),
                            )
                            pyro.sample(
                                f"{prefix}/width_{kdx}_{cdx}",
                                AffineBeta(
                                    Vindex(pyro.param(f"{prefix}/w_mean"))[
                                        kdx,
                                        ndx,
                                        fdx,
                                        cdx,
                                    ].to(self.device),
                                    Vindex(pyro.param(f"{prefix}/w_size"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device),
                                    0.75,
                                    2.25,
                                ),
                            )
                            pyro.sample(
                                f"{prefix}/x_{kdx}_{cdx}",
                                AffineBeta(
                                    Vindex(pyro.param(f"{prefix}/x_mean"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device),
                                    Vindex(pyro.param(f"{prefix}/size"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device),
                                    -(data.P + 1) / 2,
                                    (data.P + 1) / 2,
                                ),
                            )
                            pyro.sample(
                                f"{prefix}/y_{kdx}_{cdx}",
                                AffineBeta(
                                    Vindex(pyro.param(f"{prefix}/y_mean"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device),
                                    Vindex(pyro.param(f"{prefix}/size"))[
                                        kdx, ndx, fdx, cdx
                                    ].to(self.device),
                                    -(data.P + 1) / 2,
                                    (data.P + 1) / 2,
                                ),
                            )

                    pyro.sample(
                        f"{prefix}/offset_{cdx}",
                        dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                        .expand([data.P, data.P])
                        .to_event(2),
                    )

    def init_parameters(self):
        device = self.device
        pyro.param(
            "crosstalk",
            lambda: torch.tensor([0.1, 0.1], device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "proximity_loc",
            lambda: torch.tensor([0.5, 0.5], device=device),
            constraint=constraints.interval(
                0,
                (self.data.P + 1) / math.sqrt(12) - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "proximity_size",
            lambda: torch.tensor([100, 100], device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "lamda_loc",
            lambda: torch.tensor([0.5, 0.5], device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "lamda_beta",
            lambda: torch.tensor([100, 100], device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "pi_mean",
            lambda: torch.ones(self.data.C, self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "pi_size",
            lambda: torch.full((self.data.C, 1), 2, device=device),
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
                (data.N, 1, data.C),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/background_std_loc",
            lambda: torch.ones(data.N, 1, data.C, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            f"{prefix}/b_loc",
            lambda: torch.full(
                (data.N, data.F, data.C),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/b_beta",
            lambda: torch.ones(data.N, data.F, data.C, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_loc",
            lambda: torch.full((self.K, data.N, data.F, data.C), 2000, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_beta",
            lambda: torch.full((self.K, data.N, data.F, data.C), 0.001, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/w_mean",
            lambda: torch.full((self.K, data.N, data.F, data.C), 1.5, device=device),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/w_size",
            lambda: torch.full((self.K, data.N, data.F, data.C), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            f"{prefix}/x_mean",
            lambda: torch.zeros(self.K, data.N, data.F, data.C, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/y_mean",
            lambda: torch.zeros(self.K, data.N, data.F, data.C, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/size",
            lambda: torch.full((self.K, data.N, data.F, data.C), 200, device=device),
            constraint=constraints.greater_than(2.0),
        )

        # classification
        if prefix == "d":
            pyro.param(
                "d/z_probs",
                lambda: torch.ones(data.N, data.F, data.C, 1 + self.S, device=device),
                constraint=constraints.simplex,
            )
            pyro.param(
                "d/m_probs",
                lambda: torch.ones(
                    1 + self.S,
                    self.K,
                    data.N,
                    data.F,
                    data.C,
                    2,
                    device=device,
                ),
                constraint=constraints.simplex,
            )
        else:
            pyro.param(
                "c/m_probs",
                lambda: torch.ones(self.K, data.N, data.F, data.C, 2, device=device),
                constraint=constraints.simplex,
            )
