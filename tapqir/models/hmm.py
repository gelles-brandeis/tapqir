# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
hmm
^^^
"""

import math

import funsor
import torch
import torch.distributions.constraints as constraints
from pyro.distributions.hmm import _logmatmulexp
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions.utils import lazy_property
from torch.nn.functional import one_hot

from tapqir.distributions import KSMOGN, AffineBeta
from tapqir.distributions.util import expand_offtarget, probs_m, probs_theta
from tapqir.models.cosmos import cosmos


class hmm(cosmos):
    r"""
    **Multi-Color Hidden Markov Colocalization Model**

    EXPERIMENTAL This model relies on Funsor backend.

    :param K: Maximum number of spots that can be present in a single image.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    :param vectorized: Vectorize time-dimension.
    :param priors: Dictionary of parameters of prior distributions.
    """

    name = "cosmos+hmm"

    def __init__(
        self,
        K: int = 2,
        device: str = "cpu",
        dtype: str = "double",
        use_pykeops: bool = True,
        vectorized: bool = True,
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
        self.vectorized = vectorized
        super().__init__(
            K=K, device=device, dtype=dtype, use_pykeops=use_pykeops, priors=priors
        )
        self._global_params = ["gain", "proximity", "lamda", "trans"]
        self.ci_params = [
            "gain",
            "init",
            "trans",
            "lamda",
            "proximity",
            "background",
            "height",
            "width",
            "x",
            "y",
        ]

    def model(self):
        """
        **Generative Model**
        """
        # global parameters
        gain = pyro.sample("gain", dist.HalfNormal(self.priors["gain_std"]))
        init = pyro.sample(
            "init",
            dist.Dirichlet(torch.ones(self.Q, self.S + 1) / (self.S + 1)).to_event(1),
        )
        init = expand_offtarget(init)
        trans = pyro.sample(
            "trans",
            dist.Dirichlet(
                torch.ones(self.Q, self.S + 1, self.S + 1) / (self.S + 1)
            ).to_event(2),
        )
        trans = expand_offtarget(trans)
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
        frames = (
            pyro.vectorized_markov(name="frames", size=self.data.F, dim=-2)
            if self.vectorized
            else pyro.markov(range(self.data.F))
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
                z_prev = None
                for fdx in frames:
                    if self.vectorized:
                        fsx, fdx = fdx
                        fdx = torch.as_tensor(fdx)
                        fdx = fdx.unsqueeze(-1)
                    else:
                        fsx = fdx
                    # fetch data
                    obs, target_locs, is_ontarget = self.data.fetch(ndx, fdx, cdx)
                    # sample background intensity
                    background = pyro.sample(
                        f"background_f{fsx}",
                        dist.Gamma(
                            (background_mean / background_std) ** 2,
                            background_mean / background_std**2,
                        ),
                    )

                    # sample hidden model state (1+S,)
                    z_probs = (
                        Vindex(init)[..., cdx, :, is_ontarget.long()]
                        if z_prev is None
                        else Vindex(trans)[..., cdx, z_prev, :, is_ontarget.long()]
                    )
                    z_curr = pyro.sample(f"z_f{fsx}", dist.Categorical(z_probs))

                    theta = pyro.sample(
                        f"theta_f{fsx}",
                        dist.Categorical(
                            Vindex(probs_theta(self.K, self.device))[
                                torch.clamp(z_curr, min=0, max=1)
                            ]
                        ),
                        infer={"enumerate": "parallel"},
                    )
                    onehot_theta = one_hot(theta, num_classes=1 + self.K)

                    ms, heights, widths, xs, ys = [], [], [], [], []
                    for kdx in spots:
                        specific = onehot_theta[..., 1 + kdx]
                        # spot presence
                        m_probs = Vindex(probs_m(lamda, self.K))[..., cdx, theta, kdx]
                        m = pyro.sample(
                            f"m_k{kdx}_f{fsx}",
                            dist.Categorical(torch.stack((1 - m_probs, m_probs), -1)),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            height = pyro.sample(
                                f"height_k{kdx}_f{fsx}",
                                dist.HalfNormal(self.priors["height_std"]),
                            )
                            width = pyro.sample(
                                f"width_k{kdx}_f{fsx}",
                                AffineBeta(
                                    1.5,
                                    2,
                                    self.priors["width_min"],
                                    self.priors["width_max"],
                                ),
                            )
                            x = pyro.sample(
                                f"x_k{kdx}_f{fsx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_k{kdx}_f{fsx}",
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
                        f"data_f{fsx}",
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
                    z_prev = z_curr

    def guide(self):
        """
        **Variational Distribution**
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
            "init",
            dist.Dirichlet(pyro.param("init_mean") * pyro.param("init_size")).to_event(
                1
            ),
        )
        pyro.sample(
            "trans",
            dist.Dirichlet(
                pyro.param("trans_mean") * pyro.param("trans_size")
            ).to_event(2),
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
        frames = (
            pyro.vectorized_markov(name="frames", size=self.data.F, dim=-2)
            if self.vectorized
            else pyro.markov(range(self.data.F))
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
                z_prev = None
                for fdx in frames:
                    if self.vectorized:
                        fsx, fdx = fdx
                        fdx = torch.as_tensor(fdx)
                        fdx = fdx.unsqueeze(-1)
                    else:
                        fsx = fdx
                    # sample background intensity
                    pyro.sample(
                        f"background_f{fsx}",
                        dist.Gamma(
                            Vindex(pyro.param("b_loc"))[ndx, fdx, cdx]
                            * Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                        ),
                    )

                    # sample hidden model state
                    z_probs = (
                        Vindex(pyro.param("z_trans"))[ndx, fdx, cdx, 0]
                        if z_prev is None
                        else Vindex(pyro.param("z_trans"))[ndx, fdx, cdx, z_prev]
                    )
                    z_curr = pyro.sample(
                        f"z_f{fsx}",
                        dist.Categorical(z_probs),
                        infer={"enumerate": "parallel"},
                    )

                    for kdx in spots:
                        # spot presence
                        m_probs = Vindex(pyro.param("m_probs"))[
                            z_curr, kdx, ndx, fdx, cdx
                        ]
                        m = pyro.sample(
                            f"m_k{kdx}_f{fsx}",
                            dist.Categorical(torch.stack((1 - m_probs, m_probs), -1)),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"height_k{kdx}_f{fsx}",
                                dist.Gamma(
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, cdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                ),
                            )
                            pyro.sample(
                                f"width_k{kdx}_f{fsx}",
                                AffineBeta(
                                    Vindex(pyro.param("w_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx, cdx],
                                    self.priors["width_min"],
                                    self.priors["width_max"],
                                ),
                            )
                            pyro.sample(
                                f"x_k{kdx}_f{fsx}",
                                AffineBeta(
                                    Vindex(pyro.param("x_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            pyro.sample(
                                f"y_k{kdx}_f{fsx}",
                                AffineBeta(
                                    Vindex(pyro.param("y_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
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
            "init_mean",
            lambda: torch.ones(self.Q, self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "init_size",
            lambda: torch.full((self.Q, 1), 2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "trans_mean",
            lambda: torch.ones(self.Q, self.S + 1, self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "trans_size",
            lambda: torch.full((self.Q, self.S + 1, 1), 2, device=device),
            constraint=constraints.positive,
        )

        # classification
        pyro.param(
            "z_trans",
            lambda: torch.ones(
                data.Nt,
                data.F,
                data.C,
                1 + self.S,
                1 + self.S,
                device=device,
            ),
            constraint=constraints.simplex,
        )
        pyro.param(
            "m_probs",
            lambda: torch.full(
                (1 + self.S, self.K, data.Nt, data.F, data.C),
                0.5,
                device=device,
            ),
            constraint=constraints.unit_interval,
        )

        self._init_parameters()

    def TraceELBO(self, jit=False):
        """
        A trace implementation of ELBO-based SVI that supports - exhaustive enumeration over
        discrete sample sites, and - local parallel sampling over any sample site in the guide.
        """
        if self.vectorized:
            return (
                infer.JitTraceMarkovEnum_ELBO if jit else infer.TraceMarkovEnum_ELBO
            )(max_plate_nesting=3, ignore_jit_warnings=True)
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=3, ignore_jit_warnings=True
        )

    @staticmethod
    def _sequential_logmatmulexp(logits: torch.Tensor) -> torch.Tensor:
        """
        For a tensor ``x`` whose time dimension is -4, computes::
            x[..., 0, :, :, :] @ x[..., 1, :, :, :] @ ... @ x[..., T-1, :, :, :]
        but does so numerically stably in log space.
        """
        batch_shape = logits.shape[:-4]
        state_dim = logits.size(-1)
        c_dim = logits.size(-3)
        sum_terms = []
        # up sweep
        while logits.size(-4) > 1:
            time = logits.size(-4)
            even_time = time // 2 * 2
            even_part = logits[..., :even_time, :, :, :]
            x_y = even_part.reshape(
                batch_shape + (even_time // 2, 2, c_dim, state_dim, state_dim)
            )
            x, y = x_y.unbind(-4)
            contracted = _logmatmulexp(x, y)
            if time > even_time:
                contracted = torch.cat((contracted, logits[..., -1:, :, :, :]), dim=-4)
            sum_terms.append(logits)
            logits = contracted
        else:
            sum_terms.append(logits)
        # handle root case
        sum_term = sum_terms.pop()
        left_term = hmm._contraction_identity(sum_term)
        # down sweep
        while sum_terms:
            sum_term = sum_terms.pop()
            new_left_term = hmm._contraction_identity(sum_term)
            time = sum_term.size(-4)
            even_time = time // 2 * 2
            if time > even_time:
                new_left_term[..., time - 1 : time, :, :, :] = left_term[
                    ..., even_time // 2 : even_time // 2 + 1, :, :, :
                ]
                left_term = left_term[..., : even_time // 2, :, :, :]

            left_sum = sum_term[..., :even_time:2, :, :, :]
            left_sum_and_term = _logmatmulexp(left_term, left_sum)
            new_left_term[..., :even_time:2, :, :, :] = left_term
            new_left_term[..., 1:even_time:2, :, :, :] = left_sum_and_term
            left_term = new_left_term
        else:
            alphas = _logmatmulexp(left_term, sum_term)
        return alphas

    @staticmethod
    def _contraction_identity(logits: torch.Tensor) -> torch.Tensor:
        batch_shape = logits.shape[:-3]
        state_dim = logits.size(-1)
        c_dim = logits.size(-3)
        result = torch.eye(state_dim).log()
        result = result.reshape((1,) * len(batch_shape) + (1, state_dim, state_dim))
        result = result.repeat(batch_shape + (c_dim, 1, 1))
        return result

    @lazy_property
    def compute_probs(self) -> torch.Tensor:
        theta_probs = torch.zeros(self.K, self.data.Nt, self.data.F, self.Q)
        nbatch_size = self.nbatch_size
        N = sum(self.data.is_ontarget)
        for ndx in torch.split(torch.arange(N), nbatch_size):
            self.n = ndx
            self.nbatch_size = len(ndx)
            with torch.no_grad(), pyro.plate(
                "particles", size=5, dim=-4
            ), handlers.enum(first_available_dim=-5):
                guide_tr = handlers.trace(self.guide).get_trace()
                model_tr = handlers.trace(
                    handlers.replay(self.model, trace=guide_tr)
                ).get_trace()
            model_tr.compute_log_prob()
            guide_tr.compute_log_prob()

            logp = {}
            result = {}
            for fsx in ("0", f"slice(1, {self.data.F}, None)"):
                logp[fsx] = 0
                # collect log_prob terms p(z, theta, phi)
                for name in [
                    "z",
                    "theta",
                    "m_k0",
                    "m_k1",
                    "x_k0",
                    "x_k1",
                    "y_k0",
                    "y_k1",
                ]:
                    logp[fsx] += model_tr.nodes[f"{name}_f{fsx}"]["funsor"]["log_prob"]
                if fsx == "0":
                    # substitute MAP values of z into p(z=z_map, theta, phi)
                    z_map = funsor.Tensor(self.z_map[ndx, 0].long(), dtype=2)[
                        "aois", "channels"
                    ]
                    logp[fsx] = logp[fsx](**{f"z_f{fsx}": z_map})
                    # compute log_measure q for given z_map
                    log_measure = (
                        guide_tr.nodes[f"m_k0_f{fsx}"]["funsor"]["log_measure"]
                        + guide_tr.nodes[f"m_k1_f{fsx}"]["funsor"]["log_measure"]
                    )
                    log_measure = log_measure(**{f"z_f{fsx}": z_map})
                else:
                    # substitute MAP values of z into p(z=z_map, theta, phi)
                    z_map = funsor.Tensor(self.z_map[ndx, 1:].long(), dtype=2)[
                        "aois", "frames", "channels"
                    ]
                    z_map_prev = funsor.Tensor(self.z_map[ndx, :-1].long(), dtype=2)[
                        "aois", "frames", "channels"
                    ]
                    fsx_prev = f"slice(0, {self.data.F-1}, None)"
                    logp[fsx] = logp[fsx](
                        **{f"z_f{fsx}": z_map, f"z_f{fsx_prev}": z_map_prev}
                    )
                    # compute log_measure q for given z_map
                    log_measure = (
                        guide_tr.nodes[f"m_k0_f{fsx}"]["funsor"]["log_measure"]
                        + guide_tr.nodes[f"m_k1_f{fsx}"]["funsor"]["log_measure"]
                    )
                    log_measure = log_measure(
                        **{f"z_f{fsx}": z_map, f"z_f{fsx_prev}": z_map_prev}
                    )
                # compute p(z_map, theta | phi) = p(z_map, theta, phi) - p(z_map, phi)
                logp[fsx] = logp[fsx] - logp[fsx].reduce(
                    funsor.ops.logaddexp, f"theta_f{fsx}"
                )
                # average over m in p * q
                result[fsx] = (logp[fsx] + log_measure).reduce(
                    funsor.ops.logaddexp, frozenset({f"m_k0_f{fsx}", f"m_k1_f{fsx}"})
                )
                # average over particles
                result[fsx] = result[fsx].exp().reduce(funsor.ops.mean, "particles")
            theta_probs[:, ndx, 0] = result["0"].data[..., 1:].permute(2, 0, 1)
            theta_probs[:, ndx, 1:] = (
                result[f"slice(1, {self.data.F}, None)"]
                .data[..., 1:]
                .permute(3, 0, 1, 2)
            )
        self.n = None
        self.nbatch_size = nbatch_size
        return theta_probs

    @property
    def z_probs(self) -> torch.Tensor:
        r"""
        Probability of there being a target-specific spot :math:`p(z=1)`
        """
        result = self._sequential_logmatmulexp(pyro.param("z_trans").data.log())
        return result[..., 0, 1].exp()

    @property
    def theta_probs(self) -> torch.Tensor:
        r"""
        Posterior target-specific spot probability :math:`q(\theta = k, z=z_\mathsf{MAP})`.
        """
        return self.compute_probs

    @property
    def pspecific(self) -> torch.Tensor:
        r"""
        Probability of there being a target-specific spot :math:`p(\mathsf{specific})`
        """
        return self.z_probs

    @property
    def m_probs(self) -> torch.Tensor:
        r"""
        Posterior spot presence probability :math:`q(m=1, z=z_\mathsf{MAP})`.
        """
        return Vindex(torch.permute(pyro.param("m_probs").data, (1, 2, 3, 4, 0)))[
            ..., self.z_map.long()
        ]
