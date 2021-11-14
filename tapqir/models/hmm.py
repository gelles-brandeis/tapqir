# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
hmm
^^^
"""

import math
from typing import Union

import torch
import torch.distributions.constraints as constraints
from pyro.distributions.hmm import _logmatmulexp
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro

from tapqir.distributions import KSMOGN, AffineBeta
from tapqir.distributions.util import init_theta, probs_m, trans_theta
from tapqir.models.cosmos import Cosmos


class HMM(Cosmos):
    r"""
    **Single-Color Hidden Markov Colocalization Model**

    .. note::
        This model is used for kinetic simulations. Efficient fitting is not yet supported.

    **Reference**:

    1. Ordabayev YA, Friedman LJ, Gelles J, Theobald DL.
       Bayesian machine learning analysis of single-molecule fluorescence colocalization images.
       bioRxiv. 2021 Oct. doi: `10.1101/2021.09.30.462536 <https://doi.org/10.1101/2021.09.30.462536>`_.

    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :param channels: Number of color channels.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    :param vectorized: Vectorize time-dimension.
    """

    name = "hmm"

    def __init__(
        self,
        S: int = 1,
        K: int = 2,
        channels: Union[tuple, list] = (0,),
        device: str = "cpu",
        dtype: str = "double",
        use_pykeops: bool = True,
        vectorized: bool = False,
    ):
        self.vectorized = vectorized
        super().__init__(S, K, channels, device, dtype, use_pykeops)
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
        self._global_params = ["gain", "proximity", "lamda", "trans"]

    def model(self):
        # global parameters
        gain = pyro.sample("gain", dist.HalfNormal(50))
        init = pyro.sample(
            "init", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1))
        )
        trans = pyro.sample(
            "trans",
            dist.Dirichlet(torch.ones(self.S + 1, self.S + 1) / (self.S + 1)).to_event(
                1
            ),
        )
        lamda = pyro.sample("lamda", dist.Exponential(1))
        proximity = pyro.sample("proximity", dist.Exponential(1))
        size = torch.stack(
            (
                torch.full_like(proximity, 2.0),
                (((self.data.P + 1) / (2 * proximity)) ** 2 - 1),
            ),
            dim=-1,
        )

        # spots
        spots = pyro.plate("spots", self.K)
        # aoi sites
        aois = pyro.plate(
            "aois",
            self.data.N,
            subsample=self.n,
            subsample_size=self.nbatch_size,
            dim=-2,
        )
        # time frames
        frames = (
            pyro.vectorized_markov(name="frames", size=self.data.F, dim=-1)
            if self.vectorized
            else pyro.markov(range(self.data.F))
        )

        with aois as ndx:
            ndx = ndx[:, None]
            # background mean and std
            background_mean = pyro.sample("background_mean", dist.HalfNormal(1000))
            background_std = pyro.sample("background_std", dist.HalfNormal(100))
            theta_prev = None
            for fdx in frames:
                if self.vectorized:
                    fsx, fdx = fdx
                else:
                    fsx = fdx
                # fetch data
                obs, target_locs, is_ontarget = self.data.fetch(ndx, fdx, self.cdx)
                # sample background intensity
                background = pyro.sample(
                    f"background_{fdx}",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
                    ),
                )

                # sample hidden model state (1+K*S,)
                theta_probs = (
                    Vindex(init_theta(init, self.S, self.K))[..., is_ontarget.long(), :]
                    if isinstance(fdx, int) and fdx < 1
                    else Vindex(trans_theta(trans, self.S, self.K))[
                        ..., is_ontarget.long(), theta_prev, :
                    ]
                )
                theta_curr = pyro.sample(f"theta_{fsx}", dist.Categorical(theta_probs))

                ms, heights, widths, xs, ys = [], [], [], [], []
                for kdx in spots:
                    specific = Vindex(self.specific)[theta_curr, kdx]
                    # spot presence
                    m = pyro.sample(
                        f"m_{kdx}_{fsx}",
                        dist.Bernoulli(
                            Vindex(probs_m(lamda, self.S, self.K))[..., theta_curr, kdx]
                        ),
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        height = pyro.sample(
                            f"height_{kdx}_{fsx}",
                            dist.HalfNormal(10000),
                        )
                        width = pyro.sample(
                            f"width_{kdx}_{fsx}",
                            AffineBeta(
                                1.5,
                                2,
                                0.75,
                                2.25,
                            ),
                        )
                        x = pyro.sample(
                            f"x_{kdx}_{fsx}",
                            AffineBeta(
                                0,
                                Vindex(size)[..., specific],
                                -(self.data.P + 1) / 2,
                                (self.data.P + 1) / 2,
                            ),
                        )
                        y = pyro.sample(
                            f"y_{kdx}_{fsx}",
                            AffineBeta(
                                0,
                                Vindex(size)[..., specific],
                                -(self.data.P + 1) / 2,
                                (self.data.P + 1) / 2,
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
                    f"data_{fsx}",
                    KSMOGN(
                        torch.stack(heights, -1),
                        torch.stack(widths, -1),
                        torch.stack(xs, -1),
                        torch.stack(ys, -1),
                        target_locs,
                        background,
                        gain,
                        self.data.offset.samples,
                        self.data.offset.logits.to(self.dtype),
                        self.data.P,
                        torch.stack(torch.broadcast_tensors(*ms), -1),
                        self.use_pykeops,
                    ),
                    obs=obs,
                )
                theta_prev = theta_curr

    def guide(self):
        # global parameters
        pyro.sample(
            "gain",
            dist.Gamma(
                pyro.param("gain_loc") * pyro.param("gain_beta"),
                pyro.param("gain_beta"),
            ),
        )
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

        # spots
        spots = pyro.plate("spots", self.K)
        # aoi sites
        aois = pyro.plate(
            "aois",
            self.data.N,
            subsample=self.n,
            subsample_size=self.nbatch_size,
            dim=-2,
        )
        # time frames
        frames = (
            pyro.vectorized_markov(name="frames", size=self.data.F, dim=-1)
            if self.vectorized
            else pyro.markov(range(self.data.F))
        )

        with aois as ndx:
            ndx = ndx[:, None]
            pyro.sample(
                "background_mean",
                dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0]),
            )
            pyro.sample(
                "background_std",
                dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0]),
            )
            theta_prev = None
            for fdx in frames:
                if self.vectorized:
                    fsx, fdx = fdx
                else:
                    fsx = fdx
                # sample background intensity
                pyro.sample(
                    f"background_{fsx}",
                    dist.Gamma(
                        Vindex(pyro.param("b_loc"))[ndx, fdx]
                        * Vindex(pyro.param("b_beta"))[ndx, fdx],
                        Vindex(pyro.param("b_beta"))[ndx, fdx],
                    ),
                )

                # sample hidden model state (3,1,1,1)
                theta_probs = (
                    Vindex(pyro.param("theta_trans"))[ndx, fdx, 0]
                    if isinstance(fdx, int) and fdx < 1
                    else Vindex(pyro.param("theta_trans"))[ndx, fdx, theta_prev]
                )
                theta_curr = pyro.sample(
                    f"theta_{fsx}",
                    dist.Categorical(theta_probs),
                    infer={"enumerate": "parallel"},
                )

                for kdx in spots:
                    # spot presence
                    m_probs = Vindex(pyro.param("m_probs"))[theta_curr, kdx, ndx, fdx]
                    m = pyro.sample(
                        f"m_{kdx}_{fsx}",
                        dist.Categorical(m_probs),
                        infer={"enumerate": "parallel"},
                    )
                    with handlers.mask(mask=m > 0):
                        # sample spot variables
                        pyro.sample(
                            f"height_{kdx}_{fsx}",
                            dist.Gamma(
                                Vindex(pyro.param("h_loc"))[kdx, ndx, fdx]
                                * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx],
                                Vindex(pyro.param("h_beta"))[kdx, ndx, fdx],
                            ),
                        )
                        pyro.sample(
                            f"width_{kdx}_{fsx}",
                            AffineBeta(
                                Vindex(pyro.param("w_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param("w_size"))[kdx, ndx, fdx],
                                0.75,
                                2.25,
                            ),
                        )
                        pyro.sample(
                            f"x_{kdx}_{fsx}",
                            AffineBeta(
                                Vindex(pyro.param("x_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param("size"))[kdx, ndx, fdx],
                                -(self.data.P + 1) / 2,
                                (self.data.P + 1) / 2,
                            ),
                        )
                        pyro.sample(
                            f"y_{kdx}_{fsx}",
                            AffineBeta(
                                Vindex(pyro.param("y_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param("size"))[kdx, ndx, fdx],
                                -(self.data.P + 1) / 2,
                                (self.data.P + 1) / 2,
                            ),
                        )

                theta_prev = theta_curr

    def init_parameters(self):
        """
        Initialize variational parameters.
        """
        device = self.device
        data = self.data
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
        # FIX HERE
        pyro.param(
            "theta_trans",
            lambda: torch.ones(
                data.N,
                data.F,
                1 + self.K * self.S,
                1 + self.K * self.S,
                device=device,
            ),
            constraint=constraints.simplex,
        )
        m_probs = torch.full(
            (1 + self.K * self.S, self.K, self.data.N, self.data.F),
            0.5,
            device=device,
        )
        m_probs[torch.arange(self.K) + 1, torch.arange(self.K)] = 1
        pyro.param("m_probs", lambda: m_probs, constraint=constraints.unit_interval)

    def TraceELBO(self, jit=False):
        """
        A trace implementation of ELBO-based SVI that supports - exhaustive enumeration over
        discrete sample sites, and - local parallel sampling over any sample site in the guide.
        """
        if self.vectorized:
            return (
                infer.JitTraceMarkovEnum_ELBO if jit else infer.TraceMarkovEnum_ELBO
            )(max_plate_nesting=2, ignore_jit_warnings=True)
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=2, ignore_jit_warnings=True
        )

    @staticmethod
    def _sequential_logmatmulexp(logits):
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
        left_term = HMM._contraction_identity(sum_term)
        # down sweep
        while sum_terms:
            sum_term = sum_terms.pop()
            new_left_term = HMM._contraction_identity(sum_term)
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
    def theta_trans_marginal(self):
        result = self._sequential_logmatmulexp(pyro.param("theta_trans").data.log())
        return result[..., 0, :].exp()

    @property
    def theta_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        return self.theta_trans_marginal.data[..., 1:].permute(2, 0, 1)

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "sknf,nfs->knf",
            pyro.param("m_probs").data[..., 1],
            self.theta_trans_marginal,
        )
