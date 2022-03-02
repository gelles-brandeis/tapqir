# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
crosstalk
^^^^^^^^^
"""

import itertools
import math
from functools import reduce
from typing import Union

import torch
import torch.distributions.constraints as constraints
from pyro.distributions.hmm import _logmatmulexp
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property
from torch.nn.functional import one_hot

from tapqir.distributions import KSMOCTGN, AffineBeta
from tapqir.distributions.util import expand_offtarget, probs_m, probs_theta
from tapqir.models.model import Model


def cartesian_product(X, ndim=1):
    batch_shape = X.shape[: -1 - ndim]
    C = X.shape[-1 - ndim]
    S = X.shape[-1]
    result_shape = batch_shape + ndim * (S**C,)
    result = torch.zeros(result_shape)

    if ndim == 1:
        for i, s in enumerate(itertools.product(range(S), repeat=C)):
            result[..., i] = reduce(
                (lambda x, y: x * y), (X[..., c, s[c]] for c in range(C))
            )
    elif ndim == 2:
        for i, s1 in enumerate(itertools.product(range(S), repeat=C)):
            for j, s2 in enumerate(itertools.produce(range(S), repeat=C)):
                result[..., i, j] = reduce(
                    (lambda x, y: x * y), (X[..., c, s1[c], s2[c]] for c in range(C))
                )
    return result


class CrosstalkHMM(Model):
    r"""
    **Single-Color Time-Independent Colocalization Model**

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
    """

    name = "cthmm"

    def __init__(
        self,
        S: int = 1,
        K: int = 2,
        Q: int = 2,
        channels: Union[tuple, list] = (0,),
        device: str = "cpu",
        dtype: str = "double",
        use_pykeops: bool = True,
        background_mean_std: float = 1000,
        background_std_std: float = 100,
        lamda_rate: float = 1,
        height_std: float = 10000,
        width_min: float = 0.75,
        width_max: float = 2.25,
        proximity_rate: float = 1,
        gain_std: float = 50,
        vectorized: bool = True,
    ):
        self.vectorized = vectorized
        super().__init__(S, K, channels, device, dtype)
        assert S == 1, "This is a single-state model!"
        self.cdx = torch.as_tensor(self.channels)
        self.C = len(self.cdx)
        # number of fluorophore dyes
        self.Q = Q
        self.full_name = f"{self.name}-channel{self.cdx}"
        self._global_params = ["gain", "proximity", "lamda", "pi"]
        self.use_pykeops = use_pykeops
        self.conv_params = [
            "-ELBO",
            "proximity_loc_0",
            "proximity_loc_1",
            "gain_loc",
            "lamda_loc_0",
            "lamda_loc_1",
        ]
        # if S=1, C=2 => [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.z_matrix = torch.tensor(
            list(itertools.product(range(1 + self.S), repeat=self.C)), dtype=torch.long
        )

    def model(self):
        r"""
        **Generative Model**

        Model parameters:

        +-----------------+-----------+-------------------------------------+
        | Parameter       | Shape     | Description                         |
        +=================+===========+=====================================+
        | |g| - :math:`g` | (1,)      | camera gain                         |
        +-----------------+-----------+-------------------------------------+
        | |sigma| - |prox|| (1,)      | proximity                           |
        +-----------------+-----------+-------------------------------------+
        | ``lamda`` - |ld|| (1,)      | average rate of target-nonspecific  |
        |                 |           | binding                             |
        +-----------------+-----------+-------------------------------------+
        | ``pi`` - |pi|   | (1,)      | average binding probability of      |
        |                 |           | target-specific binding             |
        +-----------------+-----------+-------------------------------------+
        | |bg| - |b|      | (N, F)    | background intensity                |
        +-----------------+-----------+-------------------------------------+
        | |z| - :math:`z` | (N, F)    | target-specific spot presence       |
        +-----------------+-----------+-------------------------------------+
        | |t| - |theta|   | (N, F)    | target-specific spot index          |
        +-----------------+-----------+-------------------------------------+
        | |m| - :math:`m` | (K, N, F) | spot presence indicator             |
        +-----------------+-----------+-------------------------------------+
        | |h| - :math:`h` | (K, N, F) | spot intensity                      |
        +-----------------+-----------+-------------------------------------+
        | |w| - :math:`w` | (K, N, F) | spot width                          |
        +-----------------+-----------+-------------------------------------+
        | |x| - :math:`x` | (K, N, F) | spot position on x-axis             |
        +-----------------+-----------+-------------------------------------+
        | |y| - :math:`y` | (K, N, F) | spot position on y-axis             |
        +-----------------+-----------+-------------------------------------+
        | |D| - :math:`D` | |shape|   | observed images                     |
        +-----------------+-----------+-------------------------------------+

        .. |ps| replace:: :math:`p(\mathsf{specific})`
        .. |theta| replace:: :math:`\theta`
        .. |prox| replace:: :math:`\sigma^{xy}`
        .. |ld| replace:: :math:`\lambda`
        .. |b| replace:: :math:`b`
        .. |shape| replace:: (N, F, P, P)
        .. |sigma| replace:: ``proximity``
        .. |bg| replace:: ``background``
        .. |h| replace:: ``height``
        .. |w| replace:: ``width``
        .. |D| replace:: ``data``
        .. |m| replace:: ``m``
        .. |z| replace:: ``z``
        .. |t| replace:: ``theta``
        .. |x| replace:: ``x``
        .. |y| replace:: ``y``
        .. |pi| replace:: :math:`\pi`
        .. |g| replace:: ``gain``

        Full joint distribution:

        .. math::

            \begin{aligned}
                p(D, \phi) =~&p(g) p(\sigma^{xy}) p(\pi) p(\lambda)
                \prod_{\mathsf{AOI}} \left[ p(\mu^b) p(\sigma^b) \prod_{\mathsf{frame}}
                \left[ \vphantom{\prod_{F}} p(b | \mu^b, \sigma^b) p(z | \pi) p(\theta | z)
                \vphantom{\prod_{\substack{\mathsf{pixelX} \\ \mathsf{pixelY}}}} \cdot \right. \right. \\
                &\prod_{\mathsf{spot}} \left[ \vphantom{\prod_{F}} p(m | \theta, \lambda)
                p(h) p(w) p(x | \sigma^{xy}, \theta) p(y | \sigma^{xy}, \theta) \right] \left. \left.
                \prod_{\substack{\mathsf{pixelX} \\ \mathsf{pixelY}}} \sum_{\delta} p(\delta)
                p(D | \mu^I, g, \delta) \right] \right]
            \end{aligned}

        :math:`z` and :math:`\theta` marginalized joint distribution:

        .. math::

            \begin{aligned}
                \sum_{z, \theta} p(D, \phi) =~&p(g) p(\sigma^{xy}) p(\pi) p(\lambda)
                \prod_{\mathsf{AOI}} \left[ p(\mu^b) p(\sigma^b) \prod_{\mathsf{frame}}
                \left[ \vphantom{\prod_{F}} p(b | \mu^b, \sigma^b) \sum_{z} p(z | \pi) \sum_{\theta} p(\theta | z)
                \vphantom{\prod_{\substack{\mathsf{pixelX} \\ \mathsf{pixelY}}}} \cdot \right. \right. \\
                &\prod_{\mathsf{spot}} \left[ \vphantom{\prod_{F}} p(m | \theta, \lambda)
                p(h) p(w) p(x | \sigma^{xy}, \theta) p(y | \sigma^{xy}, \theta) \right] \left. \left.
                \prod_{\substack{\mathsf{pixelX} \\ \mathsf{pixelY}}} \sum_{\delta} p(\delta)
                p(D | \mu^I, g, \delta) \right] \right]
            \end{aligned}
        """
        # global parameters
        gain = pyro.sample("gain", dist.HalfNormal(50))
        crosstalk = pyro.sample(
            "crosstalk",
            dist.Dirichlet(torch.ones(self.Q, self.C)).to_event(1),
        )
        init = pyro.sample(
            "init",
            dist.Dirichlet(torch.ones((self.S + 1) ** self.C) / (self.S + 1) ** self.C),
        )
        init = expand_offtarget(init)
        trans = pyro.sample(
            "trans",
            dist.Dirichlet(
                torch.ones((self.S + 1) ** self.C, (self.S + 1) ** self.C)
                / (self.S + 1) ** self.C
            ).to_event(1),
        )
        trans = expand_offtarget(trans)
        lamda = pyro.sample("lamda", dist.Exponential(torch.ones(self.Q)).to_event(1))
        proximity = pyro.sample(
            "proximity", dist.Exponential(torch.ones(self.Q)).to_event(1)
        )
        size = torch.stack(
            (
                torch.full_like(proximity, 2.0),
                (((self.data.P + 1) / (2 * proximity)) ** 2 - 1),
            ),
            dim=-1,
        )

        # aoi sites
        aois = pyro.plate(
            "aois",
            self.data.Nt,
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
            ndx = ndx[:, None, None]
            # background mean and std
            background_mean = pyro.sample(
                "background_mean", dist.HalfNormal(1000).expand((self.C,)).to_event(1)
            )
            background_std = pyro.sample(
                "background_std", dist.HalfNormal(100).expand((self.C,)).to_event(1)
            )
            z_prev = None
            for fdx in frames:
                if self.vectorized:
                    fsx, fdx = fdx
                else:
                    fsx = fdx
                # fetch data
                obs, target_locs, is_ontarget = self.data.fetch(
                    ndx, torch.as_tensor(fdx).unsqueeze(-1), self.cdx
                )
                # sample background intensity
                background = pyro.sample(
                    f"background_{fsx}",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std**2,
                    ).to_event(1),
                )

                # sample hidden model state (1+S,)
                is_ontarget = is_ontarget.squeeze(-1)
                z_probs = (
                    Vindex(init)[..., :, is_ontarget.long()]
                    if isinstance(fdx, int) and fdx < 1
                    else Vindex(trans)[..., z_prev, :, is_ontarget.long()]
                )
                z_curr = pyro.sample(f"z_{fsx}", dist.Categorical(z_probs))

                ms, heights, widths, xs, ys = [], [], [], [], []
                for qdx in range(self.Q):
                    z_qdx = Vindex(self.z_matrix)[z_curr, qdx]
                    theta = pyro.sample(
                        f"theta_{qdx}_{fsx}",
                        dist.Categorical(
                            Vindex(probs_theta(self.K, self.device))[
                                torch.clamp(z_qdx, min=0, max=1)
                            ]
                        ),
                        infer={"enumerate": "parallel"},
                    )
                    onehot_theta = one_hot(theta, num_classes=1 + self.K)

                    for kdx in range(self.K):
                        specific = onehot_theta[..., 1 + kdx]
                        # spot presence
                        m_probs = Vindex(probs_m(lamda, self.K))[..., qdx, theta, kdx]
                        m = pyro.sample(
                            f"m_{kdx}_{qdx}_{fsx}",
                            dist.Categorical(torch.stack((1 - m_probs, m_probs), -1)),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            height = pyro.sample(
                                f"height_{kdx}_{qdx}_{fsx}",
                                dist.HalfNormal(10000),
                            )
                            width = pyro.sample(
                                f"width_{kdx}_{qdx}_{fsx}",
                                AffineBeta(
                                    1.5,
                                    2,
                                    0.75,
                                    2.25,
                                ),
                            )
                            x = pyro.sample(
                                f"x_{kdx}_{qdx}_{fsx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., qdx, specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_{kdx}_{qdx}_{fsx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., qdx, specific],
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

                heights = torch.stack(
                    (
                        torch.stack(heights[: self.K], -1),
                        torch.stack(heights[self.K :], -1),
                    ),
                    -2,
                )
                widths = torch.stack(
                    (
                        torch.stack(widths[: self.K], -1),
                        torch.stack(widths[self.K :], -1),
                    ),
                    -2,
                )
                xs = torch.stack(
                    (torch.stack(xs[: self.K], -1), torch.stack(xs[self.K :], -1)), -2
                )
                ys = torch.stack(
                    (torch.stack(ys[: self.K], -1), torch.stack(ys[self.K :], -1)), -2
                )
                ms = torch.broadcast_tensors(*ms)
                ms = torch.stack(
                    (torch.stack(ms[: self.K], -1), torch.stack(ms[self.K :], -1)), -2
                )
                # observed data
                pyro.sample(
                    f"data_{fsx}",
                    KSMOCTGN(
                        heights,
                        widths,
                        xs,
                        ys,
                        target_locs,
                        background,
                        gain,
                        crosstalk,
                        self.data.offset.samples,
                        self.data.offset.logits.to(self.dtype),
                        self.data.P,
                        ms,
                        self.use_pykeops,
                    ),
                    obs=obs,
                )
                z_prev = z_curr

    def guide(self):
        r"""
        **Variational Distribution**

        .. math::
            \begin{aligned}
                q(\phi \setminus \{z, \theta\}) =~&q(g) q(\sigma^{xy}) q(\pi) q(\lambda) \cdot \\
                &\prod_{\mathsf{AOI}} \left[ q(\mu^b) q(\sigma^b) \prod_{\mathsf{frame}}
                \left[ \vphantom{\prod_{F}} q(b) \prod_{\mathsf{spot}}
                q(m) q(h | m) q(w | m) q(x | m) q(y | m) \right] \right]
            \end{aligned}
        """
        # global parameters
        pyro.sample(
            "gain",
            dist.Gamma(
                pyro.param("gain_loc") * pyro.param("gain_beta"),
                pyro.param("gain_beta"),
            ),
        )
        pyro.sample("crosstalk", dist.Delta(pyro.param("crosstalk_loc")).to_event(2))
        pyro.sample(
            "init",
            dist.Dirichlet(pyro.param("init_mean") * pyro.param("init_size")),
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
            ).to_event(1),
        )
        pyro.sample(
            "proximity",
            AffineBeta(
                pyro.param("proximity_loc"),
                pyro.param("proximity_size"),
                0,
                (self.data.P + 1) / math.sqrt(12),
            ).to_event(1),
        )

        # aoi sites
        aois = pyro.plate(
            "aois",
            self.data.Nt,
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
                dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0]).to_event(
                    1
                ),
            )
            pyro.sample(
                "background_std",
                dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0]).to_event(
                    1
                ),
            )
            z_prev = None
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
                    ).to_event(1),
                )

                # sample hidden model state (3,1,1,1)
                z_probs = (
                    Vindex(pyro.param("z_trans"))[ndx, fdx, 0]
                    if isinstance(fdx, int) and fdx < 1
                    else Vindex(pyro.param("z_trans"))[ndx, fdx, z_prev]
                )
                z_curr = pyro.sample(
                    f"z_{fsx}",
                    dist.Categorical(z_probs),
                    infer={"enumerate": "parallel"},
                )

                for qdx in range(self.Q):
                    for kdx in range(self.K):
                        # sample spot presence m
                        m_probs = Vindex(pyro.param("m_probs"))[kdx, ndx, fdx, qdx]
                        m = pyro.sample(
                            f"m_{kdx}_{qdx}_{fsx}",
                            dist.Categorical(torch.stack((1 - m_probs, m_probs), -1)),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"height_{kdx}_{qdx}_{fsx}",
                                dist.Gamma(
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, qdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, qdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, qdx],
                                ),
                            )
                            pyro.sample(
                                f"width_{kdx}_{qdx}_{fsx}",
                                AffineBeta(
                                    Vindex(pyro.param("w_mean"))[kdx, ndx, fdx, qdx],
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx, qdx],
                                    0.75,
                                    2.25,
                                ),
                            )
                            pyro.sample(
                                f"x_{kdx}_{qdx}_{fsx}",
                                AffineBeta(
                                    Vindex(pyro.param("x_mean"))[kdx, ndx, fdx, qdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, qdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            pyro.sample(
                                f"y_{kdx}_{qdx}_{fsx}",
                                AffineBeta(
                                    Vindex(pyro.param("y_mean"))[kdx, ndx, fdx, qdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, qdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                z_prev = z_curr

    def init_parameters(self):
        """
        Initialize variational parameters.
        """
        device = self.device
        data = self.data
        pyro.param(
            "crosstalk_loc",
            lambda: torch.ones((self.Q, self.C), device=device) / self.C,
            constraint=constraints.simplex,
        )
        pyro.param(
            "proximity_loc",
            lambda: torch.full((self.Q,), 0.5, device=device),
            constraint=constraints.interval(
                0,
                (self.data.P + 1) / math.sqrt(12) - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "proximity_size",
            lambda: torch.full((self.Q,), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "lamda_loc",
            lambda: torch.full((self.Q,), 0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "lamda_beta",
            lambda: torch.full((self.C,), 100, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "init_mean",
            lambda: torch.ones((self.S + 1) ** self.Q, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "init_size",
            lambda: torch.full((1,), 2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "trans_mean",
            lambda: torch.ones(
                ((self.S + 1) ** self.Q, (self.S + 1) ** self.Q), device=device
            ),
            constraint=constraints.simplex,
        )
        pyro.param(
            "trans_size",
            lambda: torch.full(((self.S + 1) ** self.Q, 1), 2, device=device),
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
                (data.Nt, 1, self.C),
                data.median - data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.Nt, 1, self.C, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "b_loc",
            lambda: torch.full(
                (data.Nt, data.F, self.C),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "b_beta",
            lambda: torch.ones(data.Nt, data.F, self.C, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_loc",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 2000, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_beta",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 0.001, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "w_mean",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 1.5, device=device),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "w_size",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "x_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, self.Q, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "y_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, self.Q, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "size",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 200, device=device),
            constraint=constraints.greater_than(2.0),
        )

        # TODO fix here
        pyro.param(
            "z_trans",
            lambda: torch.ones(
                data.Nt,
                data.F,
                (1 + self.S) ** self.Q,
                (1 + self.S) ** self.Q,
                device=device,
            ),
            constraint=constraints.simplex,
        )
        pyro.param(
            "m_probs",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 0.5, device=device),
            constraint=constraints.unit_interval,
        )

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
    def _sequential_logmatmulexp(logits: torch.Tensor) -> torch.Tensor:
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
        left_term = CrosstalkHMM._contraction_identity(sum_term)
        # down sweep
        while sum_terms:
            sum_term = sum_terms.pop()
            new_left_term = CrosstalkHMM._contraction_identity(sum_term)
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
    def _contraction_identity(logits: torch.Tensor) -> torch.Tensor:
        batch_shape = logits.shape[:-2]
        state_dim = logits.size(-1)
        result = torch.eye(state_dim).log()
        result = result.reshape((1,) * len(batch_shape) + (state_dim, state_dim))
        result = result.repeat(batch_shape + (1, 1))
        return result

    @property
    def z_probs(self) -> torch.Tensor:
        r"""
        Probability of there being a target-specific spot :math:`p(z=1)`
        """
        z_probs = torch.zeros(self.data.Nt, self.data.F, self.Q)
        result = self._sequential_logmatmulexp(pyro.param("z_trans").data.log())
        z_probs[..., 0] = result[..., 0, 2].exp() + result[..., 0, 3].exp()
        z_probs[..., 1] = result[..., 0, 1].exp() + result[..., 0, 3].exp()
        return z_probs

    @property
    def theta_probs(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def m_probs(self) -> torch.Tensor:
        r"""
        Posterior spot presence probability :math:`q(m=1)`.
        """
        return pyro.param("m_probs").data

    @property
    def pspecific(self) -> torch.Tensor:
        r"""
        Probability of there being a target-specific spot :math:`p(\mathsf{specific})`
        """
        return self.z_probs

    @property
    def z_map(self) -> torch.Tensor:
        return self.z_probs > 0.5
