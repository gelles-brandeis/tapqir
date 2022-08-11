# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
crosstalk
^^^^^^^^^
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
from tapqir.models.cosmos import cosmos


class crosstalk(cosmos):
    r"""
    **Multi-Color Time-Independent Colocalization Model with Cross-Talk**

    EXPERIMENTAL

    :param K: Maximum number of spots that can be present in a single image.
    :param Q: Number of fluorescent dyes.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    :param priors: Dictionary of parameters of prior distributions.
    """

    name = "crosstalk"

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
        self._global_params = ["gain", "proximity", "lamda", "pi", "alpha"]
        self.ci_params = [
            "alpha",
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
        Generative Model
        """
        # global parameters
        gain = pyro.sample("gain", dist.HalfNormal(self.priors["gain_std"]))
        alpha = pyro.sample(
            "alpha",
            dist.Dirichlet(
                torch.ones((self.Q, self.data.C)) + torch.eye(self.Q) * 9
            ).to_event(1),
        )
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

        with aois as ndx:
            ndx = ndx[:, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                # background mean and std
                background_mean = pyro.sample(
                    "background_mean",
                    dist.HalfNormal(self.priors["background_mean_std"])
                    .expand((self.data.C,))
                    .to_event(1),
                )
                background_std = pyro.sample(
                    "background_std",
                    dist.HalfNormal(self.priors["background_std_std"])
                    .expand((self.data.C,))
                    .to_event(1),
                )
                with frames as fdx:
                    # fetch data
                    obs, target_locs, is_ontarget = self.data.fetch(
                        ndx.unsqueeze(-1), fdx.unsqueeze(-1), torch.arange(self.data.C)
                    )
                    # sample background intensity
                    background = pyro.sample(
                        "background",
                        dist.Gamma(
                            (background_mean / background_std) ** 2,
                            background_mean / background_std**2,
                        ).to_event(1),
                    )

                    ms, heights, widths, xs, ys = [], [], [], [], []
                    is_ontarget = is_ontarget.squeeze(-1)
                    for qdx in range(self.Q):
                        # sample hidden model state (1+S,)
                        z_probs = Vindex(pi)[..., qdx, :, is_ontarget.long()]
                        z = pyro.sample(
                            f"z_q{qdx}",
                            dist.Categorical(z_probs),
                            infer={"enumerate": "parallel"},
                        )
                        theta = pyro.sample(
                            f"theta_q{qdx}",
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
                                f"m_k{kdx}_q{qdx}",
                                dist.Bernoulli(
                                    Vindex(probs_m(lamda, self.K))[..., qdx, theta, kdx]
                                ),
                            )
                            with handlers.mask(mask=m > 0):
                                # sample spot variables
                                height = pyro.sample(
                                    f"height_k{kdx}_q{qdx}",
                                    dist.HalfNormal(self.priors["height_std"]),
                                )
                                width = pyro.sample(
                                    f"width_k{kdx}_q{qdx}",
                                    AffineBeta(
                                        1.5,
                                        2,
                                        self.priors["width_min"],
                                        self.priors["width_max"],
                                    ),
                                )
                                x = pyro.sample(
                                    f"x_k{kdx}_q{qdx}",
                                    AffineBeta(
                                        0,
                                        Vindex(size)[..., specific],
                                        -(self.data.P + 1) / 2,
                                        (self.data.P + 1) / 2,
                                    ),
                                )
                                y = pyro.sample(
                                    f"y_k{kdx}_q{qdx}",
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

                    heights = torch.stack(
                        [
                            torch.stack(heights[q * self.K : (1 + q) * self.K], -1)
                            for q in range(self.Q)
                        ],
                        -2,
                    )
                    widths = torch.stack(
                        [
                            torch.stack(widths[q * self.K : (1 + q) * self.K], -1)
                            for q in range(self.Q)
                        ],
                        -2,
                    )
                    xs = torch.stack(
                        [
                            torch.stack(xs[q * self.K : (1 + q) * self.K], -1)
                            for q in range(self.Q)
                        ],
                        -2,
                    )
                    ys = torch.stack(
                        [
                            torch.stack(ys[q * self.Q : (1 + q) * self.K], -1)
                            for q in range(self.Q)
                        ],
                        -2,
                    )
                    ms = torch.broadcast_tensors(*ms)
                    ms = torch.stack(
                        [
                            torch.stack(ms[q * self.Q : (1 + q) * self.K], -1)
                            for q in range(self.Q)
                        ],
                        -2,
                    )
                    # observed data
                    pyro.sample(
                        "data",
                        KSMOGN(
                            heights,
                            widths,
                            xs,
                            ys,
                            target_locs,
                            background,
                            gain,
                            self.data.offset.samples,
                            self.data.offset.logits.to(self.dtype),
                            self.data.P,
                            ms,
                            alpha,
                            use_pykeops=self.use_pykeops,
                        ),
                        obs=obs,
                    )

    def guide(self):
        r"""
        Variational Distribution
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
            "alpha",
            dist.Dirichlet(
                pyro.param("alpha_mean") * pyro.param("alpha_size")
            ).to_event(1),
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

        with aois as ndx:
            ndx = ndx[:, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                pyro.sample(
                    "background_mean",
                    dist.Delta(
                        Vindex(pyro.param("background_mean_loc"))[ndx, 0]
                    ).to_event(1),
                )
                pyro.sample(
                    "background_std",
                    dist.Delta(
                        Vindex(pyro.param("background_std_loc"))[ndx, 0]
                    ).to_event(1),
                )
                with frames as fdx:
                    # sample background intensity
                    pyro.sample(
                        "background",
                        dist.Gamma(
                            Vindex(pyro.param("b_loc"))[ndx, fdx]
                            * Vindex(pyro.param("b_beta"))[ndx, fdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx],
                        ).to_event(1),
                    )

                    for qdx in range(self.Q):
                        for kdx in range(self.K):
                            # sample spot presence m
                            m = pyro.sample(
                                f"m_k{kdx}_q{qdx}",
                                dist.Bernoulli(
                                    Vindex(pyro.param("m_probs"))[kdx, ndx, fdx, qdx]
                                ),
                                infer={"enumerate": "parallel"},
                            )
                            with handlers.mask(mask=m > 0):
                                # sample spot variables
                                pyro.sample(
                                    f"height_k{kdx}_q{qdx}",
                                    dist.Gamma(
                                        Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, qdx]
                                        * Vindex(pyro.param("h_beta"))[
                                            kdx, ndx, fdx, qdx
                                        ],
                                        Vindex(pyro.param("h_beta"))[
                                            kdx, ndx, fdx, qdx
                                        ],
                                    ),
                                )
                                pyro.sample(
                                    f"width_k{kdx}_q{qdx}",
                                    AffineBeta(
                                        Vindex(pyro.param("w_mean"))[
                                            kdx, ndx, fdx, qdx
                                        ],
                                        Vindex(pyro.param("w_size"))[
                                            kdx, ndx, fdx, qdx
                                        ],
                                        self.priors["width_min"],
                                        self.priors["width_max"],
                                    ),
                                )
                                pyro.sample(
                                    f"x_k{kdx}_q{qdx}",
                                    AffineBeta(
                                        Vindex(pyro.param("x_mean"))[
                                            kdx, ndx, fdx, qdx
                                        ],
                                        Vindex(pyro.param("size"))[kdx, ndx, fdx, qdx],
                                        -(self.data.P + 1) / 2,
                                        (self.data.P + 1) / 2,
                                    ),
                                )
                                pyro.sample(
                                    f"y_k{kdx}_q{qdx}",
                                    AffineBeta(
                                        Vindex(pyro.param("y_mean"))[
                                            kdx, ndx, fdx, qdx
                                        ],
                                        Vindex(pyro.param("size"))[kdx, ndx, fdx, qdx],
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
            "alpha_mean",
            lambda: torch.ones((self.Q, self.data.C), device=device)
            + torch.eye(self.Q, device=device) * 9,
            constraint=constraints.simplex,
        )
        pyro.param(
            "alpha_size",
            lambda: torch.full((self.Q, 1), 2, device=device),
            constraint=constraints.positive,
        )
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
        z_probs = torch.zeros(self.data.Nt, self.data.F, self.Q)
        theta_probs = torch.zeros(self.K, self.data.Nt, self.data.F, self.Q)
        nbatch_size = self.nbatch_size
        fbatch_size = self.fbatch_size
        N = sum(self.data.is_ontarget)
        params = ["m", "x", "y"]
        params = list(map(lambda x: [f"{x}_k{i}" for i in range(self.K)], params))
        params = list(itertools.chain(*params))
        params += ["z", "theta"]
        params = list(map(lambda x: [f"{x}_q{i}" for i in range(self.Q)], params))
        params = list(itertools.chain(*params))
        theta_dims = tuple(i for i in range(0, self.Q * 2, 2))
        z_dims = tuple(i for i in range(1, self.Q * 2, 2))
        m_dims = tuple(i for i in range(self.Q * 2, self.Q * (self.K + 2)))
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

                for name in params:
                    logp = logp + model_tr.nodes[name]["unscaled_log_prob"]
                # p(z, theta | phi) = p(z, theta, phi) - p(z, theta, phi).sum(z, theta)
                logp = logp - logp.logsumexp(z_dims + theta_dims)
                m_log_probs = [
                    guide_tr.nodes[f"m_k{k}_q{q}"]["unscaled_log_prob"]
                    for k in range(self.K)
                    for q in range(self.Q)
                ]
                expectation = reduce(lambda x, y: x + y, m_log_probs) + logp
                # average over m
                result = expectation.logsumexp(m_dims)
                # marginalize theta
                z_logits = result.logsumexp(theta_dims)
                a = z_logits.exp().mean(-3)
                for q in range(self.Q):
                    sum_dims = tuple(i for i in range(self.Q) if i != q)
                    if sum_dims:
                        a = a.sum(sum_dims)
                    z_probs[ndx[:, None], fdx, q] = a[1]
                # marginalize z
                b = result.logsumexp(z_dims)
                for q in range(self.Q):
                    sum_dims = tuple(i for i in range(self.Q) if i != q)
                    if sum_dims:
                        b = b.logsumexp(sum_dims)
                    theta_probs[:, ndx[:, None], fdx, q] = b[1:].exp().mean(-3)
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
