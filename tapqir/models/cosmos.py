# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
cosmos
^^^^^^
"""

import itertools
import math
from functools import reduce

import torch
import torch.distributions.constraints as constraints
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property
from torch.nn.functional import one_hot

from tapqir.distributions import KSMOGN, AffineBeta
from tapqir.distributions.util import expand_offtarget, probs_m, probs_theta
from tapqir.models.model import Model


class cosmos(Model):
    r"""
    **Multi-Color Time-Independent Colocalization Model**

    **Reference**:

    1. Ordabayev YA, Friedman LJ, Gelles J, Theobald DL.
       Bayesian machine learning analysis of single-molecule fluorescence colocalization images.
       eLife. 2022 March. doi: `10.7554/eLife.73860 <https://doi.org/10.7554/eLife.73860>`_.

    :param K: Maximum number of spots that can be present in a single image.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    :param priors: Dictionary of parameters of prior distributions.
    """

    name = "cosmos"

    def __init__(
        self,
        K: int = 2,
        Q: int = None,
        device: str = "cpu",
        dtype: str = "double",
        use_pykeops: bool = True,
        priors: dict = {
            "background_mean_std": 1000.0,
            "background_std_std": 100.0,
            "lamda_rate": 1.0,
            "height_std": 10000.0,
            "width_min": 0.75,
            "width_max": 2.25,
            "proximity_rate": 1.0,
            "gain_std": 50.0,
        },
    ):
        super().__init__(K=K, Q=Q, device=device, dtype=dtype, priors=priors)
        self._global_params = ["gain", "proximity", "lamda", "pi"]
        self.use_pykeops = use_pykeops
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
        self.ci_params = [
            "gain",
            "pi",
            "lamda",
            "proximity",
            "background",
            "height",
            "width",
            "x",
            "y",
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
        gain = pyro.sample("gain", dist.HalfNormal(self.priors["gain_std"]))
        pi = pyro.sample(
            "pi",
            dist.Dirichlet(torch.ones((self.Q, self.S + 1)) / (self.S + 1)).to_event(1),
        )
        pi = expand_offtarget(pi)
        lamda = pyro.sample(
            "lamda",
            dist.Exponential(torch.full((self.Q,), self.priors["lamda_rate"])).to_event(
                1
            ),
        )
        proximity = pyro.sample(
            "proximity", dist.Exponential(self.priors["proximity_rate"])
        )
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
            self.data.Nt,
            subsample=self.n,
            subsample_size=self.nbatch_size,
            dim=-3,
        )
        # time frames
        frames = pyro.plate(
            "frames",
            self.data.F,
            subsample=self.f,
            subsample_size=self.fbatch_size,
            dim=-2,
        )
        # color channels
        channels = pyro.plate(
            "channels",
            self.data.C,
            dim=-1,
        )

        with channels as cdx, aois as ndx:
            ndx = ndx[:, None, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                # background mean and std
                background_mean = pyro.sample(
                    "background_mean",
                    dist.HalfNormal(self.priors["background_mean_std"]),
                )
                background_std = pyro.sample(
                    "background_std", dist.HalfNormal(self.priors["background_std_std"])
                )
                with frames as fdx:
                    fdx = fdx[:, None]
                    # fetch data
                    obs, target_locs, is_ontarget = self.data.fetch(ndx, fdx, cdx)
                    # sample background intensity
                    background = pyro.sample(
                        "background",
                        dist.Gamma(
                            (background_mean / background_std) ** 2,
                            background_mean / background_std**2,
                        ),
                    )

                    # sample hidden model state (1+S,)
                    z = pyro.sample(
                        "z",
                        dist.Categorical(Vindex(pi)[..., cdx, :, is_ontarget.long()]),
                        infer={"enumerate": "parallel"},
                    )
                    theta = pyro.sample(
                        "theta",
                        dist.Categorical(
                            Vindex(probs_theta(self.K, self.device))[
                                torch.clamp(z, min=0, max=1)
                            ]
                        ),
                        infer={"enumerate": "parallel"},
                    )
                    onehot_theta = one_hot(theta, num_classes=1 + self.K)

                    ms, heights, widths, xs, ys = [], [], [], [], []
                    for kdx in spots:
                        specific = onehot_theta[..., 1 + kdx]
                        # spot presence
                        m = pyro.sample(
                            f"m_k{kdx}",
                            dist.Bernoulli(
                                Vindex(probs_m(lamda, self.K))[..., cdx, theta, kdx]
                            ),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            height = pyro.sample(
                                f"height_k{kdx}",
                                dist.HalfNormal(self.priors["height_std"]),
                            )
                            width = pyro.sample(
                                f"width_k{kdx}",
                                AffineBeta(
                                    1.5,
                                    2,
                                    self.priors["width_min"],
                                    self.priors["width_max"],
                                ),
                            )
                            x = pyro.sample(
                                f"x_k{kdx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_k{kdx}",
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
                        "data",
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
                            use_pykeops=self.use_pykeops,
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
            ),
        )

        # spots
        spots = pyro.plate("spots", self.K)
        # aoi sites
        aois = pyro.plate(
            "aois",
            self.data.Nt,
            subsample=self.n,
            subsample_size=self.nbatch_size,
            dim=-3,
        )
        # time frames
        frames = pyro.plate(
            "frames",
            self.data.F,
            subsample=self.f,
            subsample_size=self.fbatch_size,
            dim=-2,
        )
        # color channels
        channels = pyro.plate(
            "channels",
            self.data.C,
            dim=-1,
        )

        with channels as cdx, aois as ndx:
            ndx = ndx[:, None, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                pyro.sample(
                    "background_mean",
                    dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0, cdx]),
                )
                pyro.sample(
                    "background_std",
                    dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0, cdx]),
                )
                with frames as fdx:
                    fdx = fdx[:, None]
                    # sample background intensity
                    pyro.sample(
                        "background",
                        dist.Gamma(
                            Vindex(pyro.param("b_loc"))[ndx, fdx, cdx]
                            * Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                        ),
                    )

                    for kdx in spots:
                        # sample spot presence m
                        m = pyro.sample(
                            f"m_k{kdx}",
                            dist.Bernoulli(
                                Vindex(pyro.param("m_probs"))[kdx, ndx, fdx, cdx]
                            ),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"height_k{kdx}",
                                dist.Gamma(
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, cdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                ),
                            )
                            pyro.sample(
                                f"width_k{kdx}",
                                AffineBeta(
                                    Vindex(pyro.param("w_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx, cdx],
                                    self.priors["width_min"],
                                    self.priors["width_max"],
                                ),
                            )
                            pyro.sample(
                                f"x_k{kdx}",
                                AffineBeta(
                                    Vindex(pyro.param("x_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            pyro.sample(
                                f"y_k{kdx}",
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
            "pi_mean",
            lambda: torch.ones((self.Q, self.S + 1), device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "pi_size",
            lambda: torch.full((self.Q, 1), 2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "m_probs",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 0.5, device=device),
            constraint=constraints.unit_interval,
        )

        self._init_parameters()

    def _init_parameters(self):
        """
        Parameters shared between different models.
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
            lambda: torch.full((self.Q,), 0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "lamda_beta",
            lambda: torch.full((self.Q,), 100, device=device),
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
            lambda: (data.median.to(device) - data.offset.mean).expand(
                data.Nt, 1, data.C
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.Nt, 1, data.C, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "b_loc",
            lambda: (data.median.to(device) - self.data.offset.mean).expand(
                data.Nt, data.F, data.C
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "b_beta",
            lambda: torch.ones(data.Nt, data.F, data.C, device=device),
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

    def TraceELBO(self, jit=False):
        """
        A trace implementation of ELBO-based SVI that supports - exhaustive enumeration over
        discrete sample sites, and - local parallel sampling over any sample site in the guide.
        """
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=3, ignore_jit_warnings=True
        )

    @lazy_property
    def compute_probs(self) -> torch.Tensor:
        z_probs = torch.zeros(self.data.Nt, self.data.F, self.Q)
        theta_probs = torch.zeros(self.K, self.data.Nt, self.data.F, self.Q)
        nbatch_size = self.nbatch_size
        fbatch_size = self.fbatch_size
        N = sum(self.data.is_ontarget)
        params = ["m", "x", "y"]
        params = list(map(lambda x: [f"{x}_k{i}" for i in range(self.K)], params))
        params = list(itertools.chain(*params))
        params += ["z", "theta"]
        theta_dims = tuple(i for i in range(0, 2, 2))
        z_dims = tuple(i for i in range(1, 2, 2))
        m_dims = tuple(i for i in range(2, self.K + 2))
        for ndx in torch.split(torch.arange(N), nbatch_size):
            for fdx in torch.split(torch.arange(self.data.F), fbatch_size):
                self.n = ndx
                self.f = fdx
                self.nbatch_size = len(ndx)
                self.fbatch_size = len(fdx)
                qdx = torch.arange(self.Q)
                with torch.no_grad(), pyro.plate(
                    "particles", size=50, dim=-4
                ), handlers.enum(first_available_dim=-5):
                    guide_tr = handlers.trace(self.guide).get_trace()
                    model_tr = handlers.trace(
                        handlers.replay(
                            handlers.block(self.model, hide=["data"]), trace=guide_tr
                        )
                    ).get_trace()
                model_tr.compute_log_prob()
                guide_tr.compute_log_prob()
                # 0 - theta
                # 1 - z
                # 2 - m_1
                # 3 - m_0
                # p(z, theta, phi)
                logp = 0
                for name in params:
                    logp = logp + model_tr.nodes[name]["unscaled_log_prob"]
                # p(z, theta | phi) = p(z, theta, phi) - p(z, theta, phi).sum(z, theta)
                logp = logp - logp.logsumexp(z_dims + theta_dims)
                m_log_probs = [
                    guide_tr.nodes[f"m_k{k}"]["unscaled_log_prob"]
                    for k in range(self.K)
                ]
                expectation = reduce(lambda x, y: x + y, m_log_probs) + logp
                # average over m
                result = expectation.logsumexp(m_dims)
                # marginalize theta
                z_logits = result.logsumexp(theta_dims)
                z_probs[ndx[:, None, None], fdx[:, None], qdx] = (
                    z_logits[1].exp().mean(-4)
                )
                # marginalize z
                theta_logits = result.logsumexp(z_dims)
                theta_probs[:, ndx[:, None, None], fdx[:, None], qdx] = (
                    theta_logits[1:].exp().mean(-4)
                )
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
