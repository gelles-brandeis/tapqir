# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.distributions.constraints as constraints
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property

from tapqir.distributions import KSMOGN, AffineBeta
from tapqir.distributions.util import _gaussian_spots
from tapqir.models.model import Model


class Cosmos(Model):
    """
    ``cosmos`` model.

    Single-color Time-independent Colocalization Model.

    Reference:

    1. Ordabayev YA, Friedman LJ, Gelles J, Theobald DL.
       Bayesian machine learning analysis of single-molecule fluorescence colocalization images.
       bioRxiv. 2021 Oct. doi: 10.1101/2021.09.30.462536.

    :param int S: Number of distinct molecular states for the binder molecules.
    :param int K: Maximum number of spots that can be present in a single image.
    :param int channels: Number of color channels.
    :param str device: Computation device (cpu or gpu).
    :param str dtype: Floating point precision.
    :param bool use_pykeops: Use pykeops as backend to marginalize out offset.
    :param bool marginal: Marginalize out theta in the model.
    """

    name = "cosmos"

    def __init__(
        self,
        S=1,
        K=2,
        channels=(0,),
        device="cpu",
        dtype="double",
        use_pykeops=True,
        marginal=False,
    ):
        super().__init__(S, K, channels, device, dtype)
        assert S == 1, "This is a single-state model!"
        assert len(self.channels) == 1, "Please specify exactly one color channel"
        self.cdx = self.channels[0]
        self.full_name = f"{self.name}-channel{self.cdx}"
        self._global_params = ["gain", "proximity", "lamda", "pi"]
        self.use_pykeops = use_pykeops
        if marginal:
            self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
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
        # 0 (False) - offtarget
        # 1 (True) - ontarget
        result = torch.zeros(2, self.K * self.S + 1, dtype=self.dtype)
        result[0, 0] = 1
        result[1, 0] = self.pi[0]
        for s in range(self.S):
            for k in range(self.K):
                result[1, self.K * s + k + 1] = self.pi[s + 1] / self.K
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
    def theta_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        return pyro.param("theta_probs").data[..., 1:].permute(2, 0, 1)

    @property
    def j_probs(self):
        r"""
        Probability of an off-target spot :math:`p(j_{knf})`.
        """
        return self.m_probs - self.theta_probs

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "sknf,nfs->knf",
            pyro.param("m_probs").data[..., 1],
            pyro.param("theta_probs").data,
        )

    @property
    def pspecific(self):
        return self.theta_probs.sum(-3)

    @property
    def z_map(self):
        return self.pspecific > 0.5

    def model(self):
        # global parameters
        self.gain = pyro.sample("gain", dist.HalfNormal(50)).squeeze()
        self.state_model()

        # local parameters
        self.spot_model(self.data)

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
                "expose": ["theta_probs", "m_probs"],
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

            # local parameters
            self.spot_guide(self.data)

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

    def spot_model(self, data):
        # spots
        spots = pyro.plate("spots", self.K)
        # aoi sites
        aois = pyro.plate(
            "aois",
            data.N,
            subsample=self.n,
            dim=-2,
        )
        # time frames
        frames = pyro.plate(
            "frames",
            data.F,
            subsample=self.f,
            dim=-1,
        )

        with aois as ndx:
            # background mean and std
            background_mean = pyro.sample("background_mean", dist.HalfNormal(1000))
            background_std = pyro.sample("background_std", dist.HalfNormal(100))
            with frames as fdx:
                # fetch data
                obs, target_locs, is_ontarget = data.fetch(ndx[:, None], fdx, self.cdx)
                # sample background intensity
                background = pyro.sample(
                    "background",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
                    ),
                )

                # sample hidden model state (1+K*S,)
                if self._classify:
                    theta = pyro.sample(
                        "theta",
                        dist.Categorical(self.probs_theta[is_ontarget.long()]),
                    )
                else:
                    theta = pyro.sample(
                        "theta",
                        dist.Categorical(self.probs_theta[is_ontarget.long()]),
                        infer={"enumerate": "parallel"},
                    )

                ms, heights, widths, xs, ys = [], [], [], [], []
                for kdx in spots:
                    ontarget = Vindex(self.ontarget)[theta, kdx]
                    # spot presence
                    m = pyro.sample(
                        f"m_{kdx}",
                        dist.Categorical(Vindex(self.probs_m)[theta, kdx]),
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        height = pyro.sample(
                            f"height_{kdx}",
                            dist.HalfNormal(10000),
                        )
                        width = pyro.sample(
                            f"width_{kdx}",
                            AffineBeta(
                                1.5,
                                2,
                                0.75,
                                2.25,
                            ),
                        )
                        x = pyro.sample(
                            f"x_{kdx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        y = pyro.sample(
                            f"y_{kdx}",
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

                # observed data
                pyro.sample(
                    "data",
                    KSMOGN(
                        torch.stack(heights, -1),
                        torch.stack(widths, -1),
                        torch.stack(xs, -1),
                        torch.stack(ys, -1),
                        target_locs,
                        background,
                        self.gain,
                        self.data.offset.samples,
                        self.data.offset.logits.to(self.dtype),
                        data.P,
                        torch.stack(torch.broadcast_tensors(*ms), -1),
                        self.use_pykeops,
                    ),
                    obs=obs,
                )

    def spot_guide(self, data):
        # spots
        spots = pyro.plate("spots", self.K)
        # aoi sites
        aois = pyro.plate(
            "aois",
            data.N,
            subsample=self.n,
            dim=-2,
        )
        # time frames
        frames = pyro.plate(
            "frames",
            data.F,
            subsample=self.f,
            dim=-1,
        )

        with aois as ndx:
            pyro.sample(
                "background_mean",
                dist.Delta(pyro.param("background_mean_loc")[ndx].to(self.device)),
            )
            pyro.sample(
                "background_std",
                dist.Delta(pyro.param("background_std_loc")[ndx].to(self.device)),
            )
            with frames as fdx:
                # sample background intensity
                pyro.sample(
                    "background",
                    dist.Gamma(
                        Vindex(pyro.param("b_loc"))[ndx[:, None], fdx].to(self.device)
                        * Vindex(pyro.param("b_beta"))[ndx[:, None], fdx].to(
                            self.device
                        ),
                        Vindex(pyro.param("b_beta"))[ndx[:, None], fdx].to(self.device),
                    ),
                )
                if self._classify:
                    theta = pyro.sample(
                        "theta",
                        dist.Categorical(
                            Vindex(pyro.param("theta_probs"))[ndx[:, None], fdx].to(
                                self.device
                            )
                        ),
                        infer={"enumerate": "parallel"},
                    )

                for kdx in spots:
                    # sample spot presence m
                    if self._classify:
                        m_probs = Vindex(pyro.param("m_probs"))[
                            theta, kdx, ndx[:, None], fdx
                        ].to(self.device)
                    else:
                        m_probs = torch.einsum(
                            "snft,nfs->nft",
                            Vindex(pyro.param("m_probs"))[
                                torch.arange(self.S * self.K + 1)[:, None, None],
                                kdx,
                                ndx[:, None],
                                fdx,
                                :,
                            ].to(self.device),
                            Vindex(pyro.param("theta_probs"))[ndx[:, None], fdx, :].to(
                                self.device
                            ),
                        )
                    m = pyro.sample(
                        f"m_{kdx}",
                        dist.Categorical(m_probs),
                        infer={"enumerate": "parallel"},
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        pyro.sample(
                            f"height_{kdx}",
                            dist.Gamma(
                                Vindex(pyro.param("h_loc"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                )
                                * Vindex(pyro.param("h_beta"))[
                                    kdx, ndx[:, None], fdx
                                ].to(self.device),
                                Vindex(pyro.param("h_beta"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                ),
                            ),
                        )
                        pyro.sample(
                            f"width_{kdx}",
                            AffineBeta(
                                Vindex(pyro.param("w_mean"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                ),
                                Vindex(pyro.param("w_size"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                ),
                                0.75,
                                2.25,
                            ),
                        )
                        pyro.sample(
                            f"x_{kdx}",
                            AffineBeta(
                                Vindex(pyro.param("x_mean"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                ),
                                Vindex(pyro.param("size"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                ),
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        pyro.sample(
                            f"y_{kdx}",
                            AffineBeta(
                                Vindex(pyro.param("y_mean"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                ),
                                Vindex(pyro.param("size"))[kdx, ndx[:, None], fdx].to(
                                    self.device
                                ),
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
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

        self.spot_parameters(self.data)

    def spot_parameters(self, data):
        device = self.device
        pyro.param(
            "background_mean_loc",
            lambda: torch.full(
                (data.N, 1),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.N, 1, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "b_loc",
            lambda: torch.full(
                (data.N, data.F),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "b_beta",
            lambda: torch.ones(data.N, data.F, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_loc",
            lambda: torch.full((self.K, data.N, data.F), 2000, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_beta",
            lambda: torch.full((self.K, data.N, data.F), 0.001, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "w_mean",
            lambda: torch.full((self.K, data.N, data.F), 1.5, device=device),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "w_size",
            lambda: torch.full((self.K, data.N, data.F), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "x_mean",
            lambda: torch.zeros(self.K, data.N, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "y_mean",
            lambda: torch.zeros(self.K, data.N, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "size",
            lambda: torch.full((self.K, data.N, data.F), 200, device=device),
            constraint=constraints.greater_than(2.0),
        )

        # classification
        theta_probs = torch.zeros(data.N, data.F, 1 + self.K * self.S, device=device)
        theta_probs[self.data.is_ontarget] = 1 / (1 + self.K * self.S)
        theta_probs[~self.data.is_ontarget, :, 0] = 1
        pyro.param(
            "theta_probs",
            lambda: theta_probs,
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
        pyro.param("m_probs", lambda: m_probs, constraint=constraints.simplex)

    def snr(self):
        r"""
        Calculate the signal-to-noise ratio.

        Total signal:

            :math:`\mu_{knf} =  \sum_{ij} I_{nfij}
            \mathcal{N}(i, j \mid x_{knf}, y_{knf}, w_{knf})`

        Noise:

            :math:`\sigma^2_{knf} = \sigma^2_{\text{offset}}
            + \mu_{knf} \text{gain}`

        Signal-to-noise ratio:

            :math:`\text{SNR}_{knf} =
            \dfrac{\mu_{knf} - b_{nf} - \mu_{\text{offset}}}{\sigma_{knf}}
            \text{ for } \theta_{nf} = k`
        """
        weights = _gaussian_spots(
            torch.ones(1),
            self.params["d/width"]["Mean"],
            self.params["d/x"]["Mean"],
            self.params["d/y"]["Mean"],
            self.data.ontarget.xy[:, :, self.cdx, :].to(self.device),
            self.data.ontarget.P,
        )
        signal = (
            (
                self.data.ontarget.images[:, :, self.cdx, :, :]
                - self.params["d/background"]["Mean"][..., None, None]
                - self.data.offset.mean
            )
            * weights
        ).sum(dim=(-2, -1))
        noise = (
            self.data.offset.var
            + self.params["d/background"]["Mean"] * self.params["gain"]["Mean"]
        ).sqrt()
        result = signal / noise
        mask = self.theta_probs > 0.5
        return result[mask]
