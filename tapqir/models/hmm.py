# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import torch
import torch.distributions.constraints as constraints
from pyro.distributions.hmm import _logmatmulexp
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro

from tapqir import __version__ as tapqir_version
from tapqir.distributions import AffineBeta
from tapqir.models.cosmos import Cosmos


class HMM(Cosmos):
    """
    Hidden Markov model.
    """

    name = "hmm"

    def __init__(self, S=1, K=2, device="cpu", dtype="double", vectorized=True):
        self.vectorized = vectorized
        super().__init__(S, K, device, dtype)
        self.classify = True
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
        self._global_params = ["gain", "proximity", "lamda"]

    def TraceELBO(self, jit=False):
        return (infer.JitTraceMarkovEnum_ELBO if jit else infer.TraceMarkovEnum_ELBO)(
            max_plate_nesting=3, ignore_jit_warnings=True
        )

    @property
    def init_theta(self):
        result = torch.zeros(self.K * self.S + 1, dtype=self.dtype)
        result[0] = self.init[0]
        for s in range(self.S):
            for k in range(self.K):
                result[self.K * s + k + 1] = self.init[s + 1] / self.K
        return result

    @property
    def trans_theta(self):
        result = torch.zeros(self.K * self.S + 1, self.K * self.S + 1, dtype=self.dtype)
        for i in range(self.K * self.S + 1):
            # FIXME
            j = (i + 1) // self.K
            result[i, 0] = self.trans[j, 0]
            for s in range(self.S):
                for k in range(self.K):
                    result[i, self.K * s + k + 1] = self.trans[j, s + 1] / self.K
        return result

    def _sequential_logmatmulexp(self, logits):
        """
        For a tensor ``x`` whose time dimension is -3, computes::
            x[..., 0, :, :] @ x[..., 1, :, :] @ ... @ x[..., T-1, :, :]
        but does so numerically stably in log space.
        """
        batch_shape = logits.shape[:-3]
        state_dim = logits.size(-1)
        sum_terms = []
        # up sweep
        while logits.size(-3) > 1:
            time = logits.size(-3)
            even_time = time // 2 * 2
            even_part = logits[..., :even_time, :, :]
            x_y = even_part.reshape(
                batch_shape + (even_time // 2, 2, state_dim, state_dim)
            )
            x, y = x_y.unbind(-3)
            contracted = _logmatmulexp(x, y)
            if time > even_time:
                contracted = torch.cat((contracted, logits[..., -1:, :, :]), dim=-3)
            sum_terms.append(logits)
            logits = contracted
        else:
            sum_terms.append(logits)
        # handle root case
        sum_term = sum_terms.pop()
        left_term = self._contraction_identity(sum_term)
        # down sweep
        while sum_terms:
            sum_term = sum_terms.pop()
            new_left_term = self._contraction_identity(sum_term)
            time = sum_term.size(-3)
            even_time = time // 2 * 2
            if time > even_time:
                new_left_term[..., time - 1 : time, :, :] = left_term[
                    ..., even_time // 2 : even_time // 2 + 1, :, :
                ]
                left_term = left_term[..., : even_time // 2, :, :]

            left_sum = sum_term[..., :even_time:2, :, :]
            left_sum_and_term = _logmatmulexp(left_term, left_sum)
            new_left_term[..., :even_time:2, :, :] = left_term
            new_left_term[..., 1:even_time:2, :, :] = left_sum_and_term
            left_term = new_left_term
        else:
            alphas = _logmatmulexp(left_term, sum_term)
        return alphas

    @staticmethod
    def _contraction_identity(logits):
        batch_shape = logits.shape[:-2]
        state_dim = logits.size(-1)
        result = torch.eye(state_dim).log()
        result = result.reshape((1,) * len(batch_shape) + (state_dim, state_dim))
        result = result.repeat(batch_shape + (1, 1))
        return result

    @property
    def theta_probs(self):
        result = self._sequential_logmatmulexp(pyro.param("d/theta_trans").data.log())
        return result[..., 0, :].exp()

    @property
    def z_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        return self.theta_probs.data[..., 1:].permute(2, 0, 1)

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "sknf,nfs->knf", pyro.param("d/m_probs").data[..., 1], self.theta_probs
        )

    def state_model(self):
        self.init = pyro.sample(
            "init", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1))
        )
        self.trans = pyro.sample(
            "trans",
            dist.Dirichlet(torch.ones(self.S + 1, self.S + 1) / (self.S + 1)).to_event(
                1
            ),
        )
        self.lamda = pyro.sample("lamda", dist.Exponential(1)).squeeze()
        self.proximity = pyro.sample("proximity", dist.Exponential(1)).squeeze()
        self.size = torch.stack(
            (
                torch.tensor(2.0),
                (((self.data.P + 1) / (2 * self.proximity)) ** 2 - 1),
            ),
            dim=-1,
        )

    def state_guide(self):
        pyro.sample(
            "init", dist.Dirichlet(pyro.param("init_mean") * pyro.param("init_size"))
        )
        pyro.sample(
            "trans",
            dist.Dirichlet(
                pyro.param("trans_mean") * pyro.param("trans_size")
            ).to_event(1),
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

    def spot_model(self, data, prefix):
        # use time-independent model for control data
        if prefix == "c":
            return super().spot_model(data, prefix)
        # spots
        spots = pyro.plate(f"{prefix}/spots", self.K)
        # aoi sites
        aois = pyro.plate(f"{prefix}/aois", data.N, dim=-2)
        # time frames
        frames = (
            pyro.vectorized_markov(name=f"{prefix}/frames", size=data.F, dim=-1)
            if self.vectorized
            else pyro.markov(range(data.F))
        )

        with aois as ndx:
            # background mean and std
            background_mean = pyro.sample(
                f"{prefix}/background_mean", dist.HalfNormal(1000)
            )
            background_std = pyro.sample(
                f"{prefix}/background_std", dist.HalfNormal(100)
            )
            ndx = ndx[..., None]
            theta_prev = None
            for fdx in frames:
                # fetch data
                obs, target_locs = data.fetch(ndx, fdx)
                # sample background intensity
                background = pyro.sample(
                    f"{prefix}/background_{fdx}",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
                    ),
                )
                locs = background[..., None, None]

                # sample hidden model state (1+K*S,)
                if prefix == "d":
                    probs = (
                        self.init_theta
                        if isinstance(fdx, int) and fdx < 1
                        else self.trans_theta[theta_prev]
                    )
                    theta_curr = pyro.sample(
                        f"{prefix}/theta_{fdx}", dist.Categorical(probs)
                    )
                else:
                    theta_curr = 0

                for kdx in spots:
                    ontarget = Vindex(self.ontarget)[theta_curr, kdx]
                    # spot presence
                    m = pyro.sample(
                        f"{prefix}/m_{kdx}_{fdx}",
                        dist.Categorical(Vindex(self.probs_m)[theta_curr, kdx]),
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        height = pyro.sample(
                            f"{prefix}/height_{kdx}_{fdx}",
                            dist.HalfNormal(10000),
                        )
                        width = pyro.sample(
                            f"{prefix}/width_{kdx}_{fdx}",
                            AffineBeta(
                                1.5,
                                2,
                                0.75,
                                2.25,
                            ),
                        )
                        x = pyro.sample(
                            f"{prefix}/x_{kdx}_{fdx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        y = pyro.sample(
                            f"{prefix}/y_{kdx}_{fdx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )

                    # calculate image shape w/o offset
                    height = height.masked_fill(m == 0, 0)
                    gaussian = self.gaussian(height, width, x, y, target_locs)
                    locs = locs + gaussian

                # subtract offset
                odx = pyro.sample(
                    f"{prefix}/offset_{fdx}",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )
                offset = self.data.offset.samples[odx]
                offset_mask = obs > offset
                obs = torch.where(offset_mask, obs - offset, obs.new_ones(()))
                # observed data
                pyro.sample(
                    f"{prefix}/data_{fdx}",
                    dist.Gamma(
                        locs / self.gain,
                        1 / self.gain,
                    )
                    .mask(mask=offset_mask)
                    .to_event(2),
                    obs=obs,
                )
                theta_prev = theta_curr

    def spot_guide(self, data, prefix):
        # use time-independent model for control data
        if prefix == "c":
            return super().spot_guide(data, prefix)
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
        frames = (
            pyro.vectorized_markov(name=f"{prefix}/frames", size=data.F, dim=-1)
            if self.vectorized
            else pyro.markov(range(data.F))
        )

        with aois as ndx:
            if prefix == "d":
                self.batch_idx = ndx.cpu()

            pyro.sample(
                f"{prefix}/background_mean",
                dist.Delta(pyro.param(f"{prefix}/background_mean_loc")[ndx]),
            )
            pyro.sample(
                f"{prefix}/background_std",
                dist.Delta(pyro.param(f"{prefix}/background_std_loc")[ndx]),
            )
            ndx = ndx[..., None]
            theta_prev = None
            for fdx in frames:
                # sample background intensity
                pyro.sample(
                    f"{prefix}/background_{fdx}",
                    dist.Gamma(
                        Vindex(pyro.param(f"{prefix}/b_loc"))[ndx, fdx]
                        * Vindex(pyro.param(f"{prefix}/b_beta"))[ndx, fdx],
                        Vindex(pyro.param(f"{prefix}/b_beta"))[ndx, fdx],
                    ),
                )

                # sample hidden model state (3,1,1,1)
                theta_probs = (
                    Vindex(pyro.param(f"{prefix}/theta_trans"))[ndx, fdx, 0]
                    if isinstance(fdx, int) and fdx < 1
                    else Vindex(pyro.param(f"{prefix}/theta_trans"))[
                        ndx, fdx, theta_prev
                    ]
                )
                theta_curr = pyro.sample(
                    f"{prefix}/theta_{fdx}",
                    dist.Categorical(theta_probs),
                    infer={"enumerate": "parallel"},
                )

                for kdx in spots:
                    # spot presence
                    m_probs = Vindex(pyro.param(f"{prefix}/m_probs"))[
                        theta_curr, kdx, ndx, fdx
                    ]
                    m = pyro.sample(
                        f"{prefix}/m_{kdx}_{fdx}",
                        dist.Categorical(m_probs),
                        infer={"enumerate": "parallel"},
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        pyro.sample(
                            f"{prefix}/height_{kdx}_{fdx}",
                            dist.Gamma(
                                Vindex(pyro.param(f"{prefix}/h_loc"))[kdx, ndx, fdx]
                                * Vindex(pyro.param(f"{prefix}/h_beta"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/h_beta"))[kdx, ndx, fdx],
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/width_{kdx}_{fdx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/w_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/w_size"))[kdx, ndx, fdx],
                                0.75,
                                2.25,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/x_{kdx}_{fdx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/x_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/size"))[kdx, ndx, fdx],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/y_{kdx}_{fdx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/y_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/size"))[kdx, ndx, fdx],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )

                pyro.sample(
                    f"{prefix}/offset_{fdx}",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )
                theta_prev = theta_curr

    def init_parameters(self):
        # load pre-trained paramters
        self.load_checkpoint(
            path=self.path / "multispot" / tapqir_version.split("+")[0],
            param_only=True,
            warnings=True,
        )

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
        pyro.param(
            "lamda_loc", lambda: torch.tensor(0.5), constraint=constraints.positive
        )
        pyro.param(
            "lamda_beta", lambda: torch.tensor(100), constraint=constraints.positive
        )
        pyro.param(
            "init_mean", lambda: torch.ones(self.S + 1), constraint=constraints.simplex
        )
        pyro.param(
            "init_size", lambda: torch.tensor(2), constraint=constraints.positive
        )
        pyro.param(
            "trans_mean",
            lambda: torch.ones(self.S + 1, self.S + 1),
            constraint=constraints.simplex,
        )
        pyro.param(
            "trans_size",
            lambda: torch.full((self.S + 1, 1), 2),
            constraint=constraints.positive,
        )
        pyro.param(
            "d/theta_trans",
            lambda: torch.ones(
                self.data.ontarget.N,
                self.data.ontarget.F,
                1 + self.K * self.S,
                1 + self.K * self.S,
            ),
            constraint=constraints.simplex,
        )
        m_probs = torch.ones(
            1 + self.K * self.S, self.K, self.data.ontarget.N, self.data.ontarget.F, 2
        )
        m_probs[1, 0, :, :, 0] = 0
        m_probs[2, 1, :, :, 0] = 0
        pyro.param("d/m_probs", lambda: m_probs, constraint=constraints.simplex)
        if self.data.offtarget.images is not None:
            pyro.param(
                "c/m_probs",
                lambda: torch.ones(
                    self.K, self.data.offtarget.N, self.data.offtarget.F, 2
                ),
                constraint=constraints.simplex,
            )
