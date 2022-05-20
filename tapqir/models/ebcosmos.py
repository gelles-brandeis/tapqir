# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
cosmos
^^^^^^
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

from tapqir.distributions import KSMOGN, AffineBeta
from tapqir.distributions.util import expand_eye, expand_offtarget, probs_m, probs_theta
from tapqir.models.model import Model


class EBCosmos(Model):
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

    name = "ebcosmos"
    normalize_intensity = True

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
        height_std: float = 100,
        width_min: float = 0.75,
        width_max: float = 2.25,
        proximity_rate: float = 1,
        gain_std: float = 50,
    ):
        super().__init__(S, K, channels, device, dtype)
        assert S == 1, "This is a single-state model!"
        assert len(self.channels) == 1, "Please specify exactly one color channel"
        self.cdx = torch.tensor(self.channels[0])
        self.full_name = f"{self.name}-channel{self.cdx}"
        self._global_params = ["gain", "proximity", "lamda", "pi"]
        self.use_pykeops = use_pykeops
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
        # priors settings
        self.background_mean_std = background_mean_std
        self.background_std_std = background_std_std
        self.lamda_rate = lamda_rate
        self.height_std = height_std
        self.width_min = width_min
        self.width_max = width_max
        self.proximity_rate = proximity_rate
        self.gain_std = gain_std
        self.S = 4

    def model(self):
        r"""
        **Generative Model**
        """
        # global parameters
        gain = pyro.sample("gain", dist.HalfNormal(self.gain_std))
        hpi = pyro.sample("hpi", dist.Dirichlet(torch.ones(self.S) / self.S))
        hpi = expand_eye(hpi)
        lamda = pyro.sample("lamda", dist.Exponential(self.lamda_rate))
        proximity = pyro.sample("proximity", dist.Exponential(self.proximity_rate))
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

        #  height_loc = torch.cat((torch.tensor([1]), pyro.param("height_loc")), -1)
        #  height_beta = torch.cat((torch.tensor([1]), pyro.param("height_beta")), -1)

        with aois as ndx:
            ndx = ndx[:, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                pi = pyro.sample(
                    "pi",
                    dist.Dirichlet(pyro.param("rho_mean") * pyro.param("rho_size")),
                )
                pi = expand_offtarget(pi)
                height_loc = pyro.sample(
                    "height_loc",
                    dist.Gamma(
                        pyro.param("H_loc_loc") * pyro.param("H_loc_beta"),
                        pyro.param("H_loc_beta"),
                    ).to_event(1),
                )
                height_beta = pyro.sample(
                    "height_beta",
                    dist.Gamma(
                        pyro.param("H_beta_loc") * pyro.param("H_beta_beta"),
                        pyro.param("H_beta_beta"),
                    ).to_event(1),
                )
                # background mean and std
                background_mean = pyro.sample(
                    "background_mean", dist.HalfNormal(self.background_mean_std)
                )
                background_std = pyro.sample(
                    "background_std", dist.HalfNormal(self.background_std_std)
                )
                with frames as fdx:
                    # fetch data
                    obs, target_locs, is_ontarget = self.data.fetch(ndx, fdx, self.cdx)
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
                        dist.Categorical(Vindex(pi)[..., :, is_ontarget.long()]),
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
                            f"m_{kdx}",
                            dist.Bernoulli(
                                Vindex(probs_m(lamda, self.K))[..., theta, kdx]
                            ),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            h = pyro.sample(
                                f"h_{kdx}",
                                dist.Categorical(Vindex(hpi)[..., :, z * specific]),
                                infer={"enumerate": "parallel"},
                            )
                            height = pyro.sample(
                                f"height_{kdx}",
                                dist.Gamma(
                                    Vindex(height_loc)[..., h]
                                    * Vindex(height_beta)[..., h],
                                    Vindex(height_beta)[..., h],
                                ),
                            )
                            width = pyro.sample(
                                f"width_{kdx}",
                                AffineBeta(
                                    1.5,
                                    2,
                                    self.width_min,
                                    self.width_max,
                                ),
                            )
                            x = pyro.sample(
                                f"x_{kdx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_{kdx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )

                        # append
                        height = height * background
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
        pyro.sample(
            "hpi", dist.Dirichlet(pyro.param("hpi_mean") * pyro.param("hpi_size"))
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
                    "pi",
                    dist.Dirichlet(
                        Vindex(pyro.param("pi_mean"))[ndx, 0]
                        * Vindex(pyro.param("pi_size"))[ndx, 0]
                    ),
                )
                pyro.sample(
                    "height_loc",
                    dist.Delta(Vindex(pyro.param("height_loc_loc"))[ndx, 0]).to_event(
                        1
                    ),
                )
                pyro.sample(
                    "height_beta",
                    dist.Delta(Vindex(pyro.param("height_beta_loc"))[ndx, 0]).to_event(
                        1
                    ),
                )
                pyro.sample(
                    "background_mean",
                    dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0]),
                )
                pyro.sample(
                    "background_std",
                    dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0]),
                )
                with frames as fdx:
                    # sample background intensity
                    pyro.sample(
                        "background",
                        dist.Gamma(
                            Vindex(pyro.param("b_loc"))[ndx, fdx]
                            * Vindex(pyro.param("b_beta"))[ndx, fdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx],
                        ),
                    )

                    for kdx in spots:
                        # sample spot presence m
                        m = pyro.sample(
                            f"m_{kdx}",
                            dist.Bernoulli(
                                Vindex(pyro.param("m_probs"))[kdx, ndx, fdx]
                            ),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"height_{kdx}",
                                dist.Gamma(
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx],
                                ),
                            )
                            pyro.sample(
                                f"width_{kdx}",
                                AffineBeta(
                                    Vindex(pyro.param("w_mean"))[kdx, ndx, fdx],
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx],
                                    0.75,
                                    2.25,
                                ),
                            )
                            pyro.sample(
                                f"x_{kdx}",
                                AffineBeta(
                                    Vindex(pyro.param("x_mean"))[kdx, ndx, fdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            pyro.sample(
                                f"y_{kdx}",
                                AffineBeta(
                                    Vindex(pyro.param("y_mean"))[kdx, ndx, fdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx],
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

        checkpoint = torch.load(
            self.run_path / f"cosmos2-channel{self.cdx}-model.tpqr", map_location=device
        )
        del checkpoint["params"]["params"]["pi_mean"]
        del checkpoint["params"]["params"]["pi_size"]
        del checkpoint["params"]["constraints"]["pi_mean"]
        del checkpoint["params"]["constraints"]["pi_size"]
        pyro.get_param_store().set_state(checkpoint["params"])

        pyro.param(
            "H_loc_loc",
            lambda: torch.tensor([10, 20, 30, 40], device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "H_loc_beta",
            lambda: torch.full((self.S,), 0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "H_beta_loc",
            lambda: torch.full((self.S,), 0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "H_beta_beta",
            lambda: torch.full((self.S,), 0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "height_loc_loc",
            lambda: torch.tensor([10, 20, 30, 40], device=device).expand(
                data.Nt, 1, self.S
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "height_beta_loc",
            lambda: torch.full(
                (
                    data.Nt,
                    1,
                    self.S,
                ),
                0.5,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "rho_mean",
            lambda: torch.ones(self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "rho_size",
            lambda: torch.tensor(2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "pi_mean",
            lambda: torch.ones((data.Nt, 1, self.S + 1), device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "pi_size",
            lambda: torch.full((data.Nt, 1, 1), 2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "hpi_mean",
            lambda: torch.ones(self.S, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "hpi_size",
            lambda: torch.tensor(2, device=device),
            constraint=constraints.positive,
        )

        # self._init_parameters()

    def _init_parameters(self):
        """
        Parameters shared between different models.
        """
        device = self.device
        data = self.data

        pyro.param(
            "height_std",
            lambda: torch.tensor(100, device=device),
            constraint=constraints.positive,
        )
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
                (data.Nt, 1),
                data.median[self.cdx] - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.Nt, 1, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "b_loc",
            lambda: torch.full(
                (data.Nt, data.F),
                data.median[self.cdx] - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "b_beta",
            lambda: torch.ones(data.Nt, data.F, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_loc",
            lambda: torch.full((self.K, data.Nt, data.F), 200, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_beta",
            lambda: torch.full((self.K, data.Nt, data.F), 0.01, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "w_mean",
            lambda: torch.full((self.K, data.Nt, data.F), 1.5, device=device),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "w_size",
            lambda: torch.full((self.K, data.Nt, data.F), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "x_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "y_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "size",
            lambda: torch.full((self.K, data.Nt, data.F), 200, device=device),
            constraint=constraints.greater_than(2.0),
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
        z_probs = torch.zeros(self.data.Nt, self.data.F, self.S + 1)
        theta_probs = torch.zeros(self.K, self.data.Nt, self.data.F)
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
                    "particles", size=50, dim=-3
                ), handlers.enum(first_available_dim=-4):
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
                for name in [
                    "z",
                    "theta",
                    "h_0",
                    "h_1",
                    "m_0",
                    "m_1",
                    "height_0",
                    "height_1",
                    "x_0",
                    "x_1",
                    "y_0",
                    "y_1",
                ]:
                    logp = logp + model_tr.nodes[name]["unscaled_log_prob"]
                # p(z, theta | phi) = p(z, theta, phi) - p(z, theta, phi).sum(z, theta)
                # breakpoint()
                logp = logp.logsumexp((0, 1))
                logp = logp - logp.logsumexp((0, 1))
                expectation = (
                    guide_tr.nodes["m_0"]["unscaled_log_prob"]
                    + guide_tr.nodes["m_1"]["unscaled_log_prob"]
                    + logp
                )
                # average over m
                result = expectation.logsumexp((2, 3))
                # marginalize theta
                z_logits = result.logsumexp(0)
                z_probs[ndx[:, None], fdx, :] = z_logits.exp().mean(1).permute(1, 2, 0)
                # marginalize z
                theta_logits = result.logsumexp(1)
                theta_probs[:, ndx[:, None], fdx] = theta_logits[1:].exp().mean(-3)
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
        return self.theta_probs.sum(0)

    @property
    def z_map(self) -> torch.Tensor:
        return self.z_probs.argmax(-1)
