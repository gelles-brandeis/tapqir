# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.distributions.constraints as constraints
from pyro.distributions.hmm import _logmatmulexp
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro

from tapqir.distributions import AffineBeta, KSpotGammaNoise
from tapqir.models.cosmos import Cosmos


class HMM(Cosmos):
    """
    Hidden Markov model.
    """

    name = "hmmz"

    def __init__(
        self, S=1, K=2, device="cpu", dtype="double", marginal=False, vectorized=True
    ):
        self.vectorized = vectorized
        super().__init__(S, K, device, dtype)
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
        self._global_params = ["gain", "proximity", "lamda", "trans"]

    def TraceELBO(self, jit=False):
        if self.vectorized:
            return (
                infer.JitTraceMarkovEnum_ELBO if jit else infer.TraceMarkovEnum_ELBO
            )(max_plate_nesting=2, ignore_jit_warnings=True)
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=2, ignore_jit_warnings=True
        )

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
    def pspecific(self):
        result = self._sequential_logmatmulexp(pyro.param("d/z_trans").data.log())
        return result[..., 0, 1].exp()

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

    def guide(self):
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
            z_prev = None
            for fdx in frames:
                if self.vectorized:
                    fsx, fdx = fdx
                else:
                    fsx = fdx
                # sample background intensity
                background = pyro.sample(
                    f"{prefix}/background_{fsx}",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
                    ),
                )

                # sample hidden model state (1+S,)
                probs = (
                    self.init
                    if isinstance(fdx, int) and fdx < 1
                    else self.trans[z_prev]
                )
                z_curr = pyro.sample(f"{prefix}/z_{fsx}", dist.Categorical(probs))
                theta = pyro.sample(
                    f"{prefix}/theta_{fsx}",
                    dist.Categorical(self.probs_theta[z_curr]),
                    infer={"enumerate": "parallel"},
                )

                ms, heights, widths, xs, ys = [], [], [], [], []
                for kdx in spots:
                    ontarget = Vindex(self.ontarget)[theta, kdx]
                    # spot presence
                    m = pyro.sample(
                        f"{prefix}/m_{kdx}_{fsx}",
                        dist.Categorical(Vindex(self.probs_m)[theta, kdx]),
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        height = pyro.sample(
                            f"{prefix}/height_{kdx}_{fsx}",
                            dist.HalfNormal(10000),
                        )
                        width = pyro.sample(
                            f"{prefix}/width_{kdx}_{fsx}",
                            AffineBeta(
                                1.5,
                                2,
                                0.75,
                                2.25,
                            ),
                        )
                        x = pyro.sample(
                            f"{prefix}/x_{kdx}_{fsx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        y = pyro.sample(
                            f"{prefix}/y_{kdx}_{fsx}",
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
                    f"{prefix}/offset_{fsx}",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )
                offset = self.data.offset.samples[odx]
                # fetch data
                obs, target_locs = data.fetch(ndx, fdx)
                # observed data
                pyro.sample(
                    f"{prefix}/data_{fsx}",
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
                z_prev = z_curr

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
            pyro.sample(
                f"{prefix}/background_mean",
                dist.Delta(pyro.param(f"{prefix}/background_mean_loc")[ndx]),
            )
            pyro.sample(
                f"{prefix}/background_std",
                dist.Delta(pyro.param(f"{prefix}/background_std_loc")[ndx]),
            )
            ndx = ndx[..., None]
            z_prev = None
            for fdx in frames:
                if self.vectorized:
                    fsx, fdx = fdx
                else:
                    fsx = fdx
                # sample background intensity
                pyro.sample(
                    f"{prefix}/background_{fsx}",
                    dist.Gamma(
                        Vindex(pyro.param(f"{prefix}/b_loc"))[ndx, fdx]
                        * Vindex(pyro.param(f"{prefix}/b_beta"))[ndx, fdx],
                        Vindex(pyro.param(f"{prefix}/b_beta"))[ndx, fdx],
                    ),
                )

                # sample hidden model state (3,1,1,1)
                z_probs = (
                    Vindex(pyro.param(f"{prefix}/z_trans"))[ndx, fdx, 0]
                    if isinstance(fdx, int) and fdx < 1
                    else Vindex(pyro.param(f"{prefix}/z_trans"))[ndx, fdx, z_prev]
                )
                z_curr = pyro.sample(
                    f"{prefix}/z_{fsx}",
                    dist.Categorical(z_probs),
                    infer={"enumerate": "parallel"},
                )

                for kdx in spots:
                    # spot presence
                    m_probs = Vindex(pyro.param(f"{prefix}/m_probs"))[
                        z_curr, kdx, ndx, fdx
                    ]
                    m = pyro.sample(
                        f"{prefix}/m_{kdx}_{fsx}",
                        dist.Categorical(m_probs),
                        infer={"enumerate": "parallel"},
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        pyro.sample(
                            f"{prefix}/height_{kdx}_{fsx}",
                            dist.Gamma(
                                Vindex(pyro.param(f"{prefix}/h_loc"))[kdx, ndx, fdx]
                                * Vindex(pyro.param(f"{prefix}/h_beta"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/h_beta"))[kdx, ndx, fdx],
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/width_{kdx}_{fsx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/w_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/w_size"))[kdx, ndx, fdx],
                                0.75,
                                2.25,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/x_{kdx}_{fsx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/x_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/size"))[kdx, ndx, fdx],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/y_{kdx}_{fsx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/y_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/size"))[kdx, ndx, fdx],
                                -(data.P + 1) / 2,
                                (data.P + 1) / 2,
                            ),
                        )

                pyro.sample(
                    f"{prefix}/offset_{fsx}",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )
                z_prev = z_curr

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
            "init_mean",
            lambda: torch.ones(self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "init_size",
            lambda: torch.tensor(2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "trans_mean",
            lambda: torch.ones(self.S + 1, self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "trans_size",
            lambda: torch.full((self.S + 1, 1), 2, device=device),
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
                "d/z_trans",
                lambda: torch.ones(
                    data.N,
                    data.F,
                    1 + self.S,
                    1 + self.S,
                    device=device,
                ),
                constraint=constraints.simplex,
            )
            pyro.param(
                "d/m_probs",
                lambda: torch.ones(
                    1 + self.S,
                    self.K,
                    data.N,
                    data.F,
                    2,
                    device=device,
                ),
                constraint=constraints.simplex,
            )
        else:
            pyro.param(
                "c/m_probs",
                lambda: torch.ones(self.K, data.N, data.F, 2, device=device),
                constraint=constraints.simplex,
            )
