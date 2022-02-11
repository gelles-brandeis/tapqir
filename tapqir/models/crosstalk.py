# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
crosstalk
^^^^^^^^^
"""

import math
from typing import Union

import torch
import torch.distributions.constraints as constraints
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property
from torch.nn.functional import one_hot

from tapqir.distributions import KSMOCTGN, AffineBeta
from tapqir.distributions.util import expand_offtarget, probs_m, probs_theta
from tapqir.models.model import Model


class Crosstalk(Model):
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

    name = "crosstalk"

    def __init__(
        self,
        S: int = 1,
        K: int = 2,
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
    ):
        super().__init__(S, K, channels, device, dtype)
        assert S == 1, "This is a single-state model!"
        self.cdx = torch.tensor(self.channels)
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
            "crosstalk_sample",
            dist.HalfNormal(torch.full((self.data.C,), 0.5)).to_event(1),
        )
        pi = pyro.sample(
            "pi",
            dist.Dirichlet(torch.ones(self.data.C, self.S + 1) / (self.S + 1)).to_event(
                1
            ),
        )
        pi = expand_offtarget(pi)
        lamda = pyro.sample(
            "lamda", dist.Exponential(torch.ones(self.data.C)).to_event(1)
        )
        proximity = pyro.sample(
            "proximity", dist.Exponential(torch.ones(self.data.C)).to_event(1)
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
        frames = pyro.plate(
            "frames",
            self.data.F,
            subsample=self.f,
            subsample_size=self.fbatch_size,
            dim=-1,
        )

        with aois as ndx, frames as fdx:
            ndx = ndx[:, None, None]
            fdx = fdx[:, None]
            # fetch data
            obs, target_locs, is_ontarget = self.data.fetch(ndx, fdx, self.cdx)
            is_ontarget = is_ontarget[..., 0]

        ms, heights, widths, xs, ys, backgrounds = [], [], [], [], [], []
        for cdx in range(self.data.C):
            with aois as ndx:
                ndx = ndx[:, None]
                # background mean and std
                background_mean = pyro.sample(
                    f"background_mean_{cdx}", dist.HalfNormal(1000)
                )
                background_std = pyro.sample(
                    f"background_std_{cdx}", dist.HalfNormal(100)
                )
                with frames as fdx:
                    # sample background intensity
                    background = pyro.sample(
                        f"background_{cdx}",
                        dist.Gamma(
                            (background_mean / background_std) ** 2,
                            background_mean / background_std**2,
                        ),
                    )
                    backgrounds.append(background)

                    # sample hidden model state (1+S,)
                    z = pyro.sample(
                        f"z_{cdx}",
                        dist.Categorical(Vindex(pi)[..., cdx, :, is_ontarget.long()]),
                        infer={"enumerate": "parallel"},
                    )
                    theta = pyro.sample(
                        f"theta_{cdx}",
                        dist.Categorical(
                            Vindex(probs_theta(self.K, self.device))[
                                torch.clamp(z, min=0, max=1)
                            ]
                        ),
                        infer={"enumerate": "parallel"},
                    )
                    onehot_theta = one_hot(theta, num_classes=1 + self.K)

                    for kdx in range(self.K):
                        specific = onehot_theta[..., 1 + kdx]
                        # spot presence
                        m = pyro.sample(
                            f"m_{kdx}_{cdx}",
                            dist.Bernoulli(
                                Vindex(probs_m(lamda, self.K))[..., cdx, theta, kdx]
                            ),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            height = pyro.sample(
                                f"height_{kdx}_{cdx}",
                                dist.HalfNormal(10000),
                            )
                            width = pyro.sample(
                                f"width_{kdx}_{cdx}",
                                AffineBeta(
                                    1.5,
                                    2,
                                    0.75,
                                    2.25,
                                ),
                            )
                            x = pyro.sample(
                                f"x_{kdx}_{cdx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., cdx, specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_{kdx}_{cdx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., cdx, specific],
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

        with aois as ndx, frames as fdx:
            heights = torch.stack(
                (
                    torch.stack(heights[: self.K], -1),
                    torch.stack(heights[self.K :], -1),
                ),
                -2,
            )
            widths = torch.stack(
                (torch.stack(widths[: self.K], -1), torch.stack(widths[self.K :], -1)),
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
            backgrounds = torch.stack(backgrounds, -1)
            # observed data
            pyro.sample(
                "data",
                KSMOCTGN(
                    heights,
                    widths,
                    xs,
                    ys,
                    target_locs,
                    backgrounds,
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
        pyro.sample("crosstalk_sample", dist.Delta(pyro.param("crosstalk")).to_event(1))
        pyro.sample(
            "pi",
            dist.Dirichlet(pyro.param("pi_mean") * pyro.param("pi_size")).to_event(1),
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
        frames = pyro.plate(
            "frames",
            self.data.F,
            subsample=self.f,
            subsample_size=self.fbatch_size,
            dim=-1,
        )

        for cdx in range(self.data.C):
            with aois as ndx:
                ndx = ndx[:, None]
                pyro.sample(
                    f"background_mean_{cdx}",
                    dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0, cdx]),
                )
                pyro.sample(
                    f"background_std_{cdx}",
                    dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0, cdx]),
                )
                with frames as fdx:
                    # sample background intensity
                    pyro.sample(
                        f"background_{cdx}",
                        dist.Gamma(
                            Vindex(pyro.param("b_loc"))[ndx, fdx, cdx]
                            * Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                        ),
                    )

                    for kdx in range(self.K):
                        # sample spot presence m
                        m = pyro.sample(
                            f"m_{kdx}_{cdx}",
                            dist.Bernoulli(
                                Vindex(pyro.param("m_probs"))[kdx, ndx, fdx, cdx]
                            ),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"height_{kdx}_{cdx}",
                                dist.Gamma(
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, cdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                ),
                            )
                            pyro.sample(
                                f"width_{kdx}_{cdx}",
                                AffineBeta(
                                    Vindex(pyro.param("w_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx, cdx],
                                    0.75,
                                    2.25,
                                ),
                            )
                            pyro.sample(
                                f"x_{kdx}_{cdx}",
                                AffineBeta(
                                    Vindex(pyro.param("x_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            pyro.sample(
                                f"y_{kdx}_{cdx}",
                                AffineBeta(
                                    Vindex(pyro.param("y_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )

    def init_parameters(self):
        """
        Initialize variational parameters.
        """
        device = self.device
        data = self.data
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
            lambda: torch.ones((self.data.C, self.S + 1), device=device),
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

        pyro.param(
            "background_mean_loc",
            lambda: torch.full(
                (data.Nt, 1, self.data.C),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.Nt, 1, self.data.C, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "b_loc",
            lambda: torch.full(
                (data.Nt, data.F, self.data.C),
                data.median - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "b_beta",
            lambda: torch.ones(data.Nt, data.F, self.data.C, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_loc",
            lambda: torch.full(
                (self.K, data.Nt, data.F, self.data.C), 2000, device=device
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_beta",
            lambda: torch.full(
                (self.K, data.Nt, data.F, self.data.C), 0.001, device=device
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "w_mean",
            lambda: torch.full(
                (self.K, data.Nt, data.F, self.data.C), 1.5, device=device
            ),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "w_size",
            lambda: torch.full(
                (self.K, data.Nt, data.F, self.data.C), 100, device=device
            ),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "x_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, self.data.C, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "y_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, self.data.C, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "size",
            lambda: torch.full(
                (self.K, data.Nt, data.F, self.data.C), 200, device=device
            ),
            constraint=constraints.greater_than(2.0),
        )

        pyro.param(
            "m_probs",
            lambda: torch.full(
                (self.K, data.Nt, data.F, self.data.C), 0.5, device=device
            ),
            constraint=constraints.unit_interval,
        )

    def TraceELBO(self, jit=False):
        """
        A trace implementation of ELBO-based SVI that supports - exhaustive enumeration over
        discrete sample sites, and - local parallel sampling over any sample site in the guide.
        """
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=2, ignore_jit_warnings=True
        )

    @lazy_property
    def compute_probs(self) -> torch.Tensor:
        z_probs = torch.zeros(self.data.Nt, self.data.F, self.data.C)
        theta_probs = torch.zeros(self.K, self.data.Nt, self.data.F, self.data.C)
        nbatch_size = self.nbatch_size
        fbatch_size = self.fbatch_size
        N = sum(self.data.is_ontarget)
        for ndx in torch.split(torch.arange(N), nbatch_size):
            for fdx in torch.split(torch.arange(self.data.F), fbatch_size):
                self.n = ndx
                self.f = fdx
                self.nbatch_size = len(ndx)
                self.fbatch_size = len(fdx)
                with torch.no_grad(), pyro.plate(
                    "particles", size=5, dim=-3
                ), handlers.enum(first_available_dim=-4):
                    guide_tr = handlers.trace(self.guide).get_trace()
                    model_tr = handlers.trace(
                        handlers.replay(self.model, trace=guide_tr)
                    ).get_trace()
                model_tr.compute_log_prob()
                guide_tr.compute_log_prob()
                # 0 - theta
                # 1 - z
                # 2 - m_1
                # 3 - m_0
                # p(z, theta, phi)
                logp = 0
                for name in [
                    "z_0",
                    "z_1",
                    "theta_0",
                    "theta_1",
                    "m_0_0",
                    "m_0_1",
                    "m_1_0",
                    "m_1_1",
                    "x_0_0",
                    "x_0_1",
                    "x_1_0",
                    "x_1_1",
                    "y_0_0",
                    "y_0_1",
                    "y_1_0",
                    "y_1_1",
                ]:
                    logp = logp + model_tr.nodes[name]["unscaled_log_prob"]
                # p(z, theta | phi) = p(z, theta, phi) - p(z, theta, phi).sum(z, theta)
                logp = logp - logp.logsumexp((0, 1, 2, 3))
                expectation = (
                    guide_tr.nodes["m_0_0"]["unscaled_log_prob"]
                    + guide_tr.nodes["m_0_1"]["unscaled_log_prob"]
                    + guide_tr.nodes["m_1_0"]["unscaled_log_prob"]
                    + guide_tr.nodes["m_1_1"]["unscaled_log_prob"]
                    + logp
                )
                # average over m
                result = expectation.logsumexp((4, 5, 6, 7))
                # marginalize theta
                z_logits = result.logsumexp((0, 1, 2))
                z_probs[ndx[:, None], fdx, 0] = z_logits[1].exp().mean(-3)
                z_logits = result.logsumexp((0, 2, 3))
                z_probs[ndx[:, None], fdx, 1] = z_logits[1].exp().mean(-3)
                # marginalize z
                theta_logits = result.logsumexp((0, 1, 3))
                theta_probs[:, ndx[:, None], fdx, 0] = theta_logits[1:].exp().mean(-3)
                theta_logits = result.logsumexp((1, 2, 3))
                theta_probs[:, ndx[:, None], fdx, 1] = theta_logits[1:].exp().mean(-3)
        self.n = None
        self.f = None
        self.nbatch_size = nbatch_size
        self.fbatch_size = fbatch_size
        return z_probs, theta_probs

    @property
    def z_probs(self) -> torch.Tensor:
        r"""
        Probability of there being a target-specific spot :math:`p(z=1)`
        """
        return self.compute_probs[0]

    @property
    def theta_probs(self) -> torch.Tensor:
        r"""
        Posterior target-specific spot probability :math:`q(\theta = k)`.
        """
        return self.compute_probs[1]

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
