# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

"""
hmm
^^^
"""

import math
from typing import Union

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
from tapqir.distributions.util import expand_eye, expand_offtarget, probs_m, probs_theta

from .hmm import HMM


class MSHMM(HMM):
    r"""
    **Single-Color Hidden Markov Colocalization Model**

    EXPERIMENTAL This model relies on Funsor backend.

    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :param channels: Number of color channels.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    :param vectorized: Vectorize time-dimension.
    """

    name = "mshmm"
    normalize_intensity = True

    def __init__(
        self,
        S: int = 1,
        K: int = 2,
        channels: Union[tuple, list] = (0,),
        device: str = "cpu",
        dtype: str = "double",
        use_pykeops: bool = True,
        vectorized: bool = True,
        background_mean_std: float = 1000,
        background_std_std: float = 100,
        lamda_rate: float = 1,
        height_std: float = 10000,
        width_min: float = 0.75,
        width_max: float = 2.25,
        proximity_rate: float = 1,
        gain_std: float = 50,
    ):
        self.vectorized = vectorized
        super().__init__(S, K, channels, device, dtype, use_pykeops)
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
        self._global_params = ["gain", "proximity", "lamda", "trans"]
        self.S = 4

    def model(self):
        """
        **Generative Model**
        """
        # global parameters
        gain = pyro.sample("gain", dist.HalfNormal(50))
        init = pyro.sample(
            "init", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1))
        )
        init = expand_offtarget(init)
        trans = pyro.sample(
            "trans",
            dist.Dirichlet(torch.ones(self.S + 1, self.S + 1) / (self.S + 1)).to_event(
                1
            ),
        )
        trans = expand_offtarget(trans)
        hpi = pyro.sample("hpi", dist.Dirichlet(torch.ones(self.S) / self.S))
        hpi = expand_eye(hpi)
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
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                # background mean and std
                background_mean = pyro.sample("background_mean", dist.HalfNormal(1000))
                background_std = pyro.sample("background_std", dist.HalfNormal(100))
                z_prev = None
                for fdx in frames:
                    if self.vectorized:
                        fsx, fdx = fdx
                    else:
                        fsx = fdx
                    # fetch data
                    obs, target_locs, is_ontarget = self.data.fetch(ndx, fdx, self.cdx)
                    # sample background intensity
                    background = pyro.sample(
                        f"background_{fsx}",
                        dist.Gamma(
                            (background_mean / background_std) ** 2,
                            background_mean / background_std**2,
                        ),
                    )

                    # sample hidden model state (1+S,)
                    z_probs = (
                        Vindex(init)[..., :, is_ontarget.long()]
                        if isinstance(fdx, int) and fdx < 1
                        else Vindex(trans)[..., z_prev, :, is_ontarget.long()]
                    )
                    z_curr = pyro.sample(f"z_{fsx}", dist.Categorical(z_probs))

                    theta = pyro.sample(
                        f"theta_{fsx}",
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
                        m_probs = Vindex(probs_m(lamda, self.K))[..., theta, kdx]
                        m = pyro.sample(
                            f"m_{kdx}_{fsx}",
                            dist.Categorical(torch.stack((1 - m_probs, m_probs), -1)),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            h = pyro.sample(
                                f"h_{kdx}_{fsx}",
                                dist.Categorical(
                                    Vindex(hpi)[..., :, z_curr * specific]
                                ),
                                infer={"enumerate": "parallel"},
                            )
                            height = pyro.sample(
                                f"height_{kdx}_{fsx}",
                                dist.Gamma(
                                    pyro.param("height_loc")[h]
                                    * pyro.param("height_beta")[h],
                                    pyro.param("height_beta")[h],
                                ),
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
                        height = height * background
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
            "init", dist.Dirichlet(pyro.param("init_mean") * pyro.param("init_size"))
        )
        pyro.sample(
            "trans",
            dist.Dirichlet(
                pyro.param("trans_mean") * pyro.param("trans_size")
            ).to_event(1),
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
        frames = (
            pyro.vectorized_markov(name="frames", size=self.data.F, dim=-1)
            if self.vectorized
            else pyro.markov(range(self.data.F))
        )

        with aois as ndx:
            ndx = ndx[:, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                pyro.sample(
                    "background_mean",
                    dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0]),
                )
                pyro.sample(
                    "background_std",
                    dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0]),
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
                        ),
                    )

                    # sample hidden model state
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

                    for kdx in spots:
                        # spot presence
                        m_probs = Vindex(pyro.param("m_probs"))[kdx, ndx, fdx]
                        m = pyro.sample(
                            f"m_{kdx}_{fsx}",
                            dist.Categorical(torch.stack((1 - m_probs, m_probs), -1)),
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

                    z_prev = z_curr

    def init_parameters(self):
        """
        Initialize variational parameters.
        """
        device = self.device
        data = self.data

        checkpoint = torch.load(
            self.run_path / f"mscosmos-channel{self.cdx}-model.tpqr",
            map_location=device,
        )
        params = torch.load(self.path / f"mscosmos-channel{self.cdx}-params.tpqr")
        del checkpoint["params"]["params"]["pi_mean"]
        del checkpoint["params"]["params"]["pi_size"]
        del checkpoint["params"]["constraints"]["pi_mean"]
        del checkpoint["params"]["constraints"]["pi_size"]
        pyro.get_param_store().set_state(checkpoint["params"])

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

        # classification
        params["z_probs"][data.N :, :, 0] = 1
        z_trans = params["z_probs"]  # .log()
        z_trans = z_trans.unsqueeze(-2).expand(data.Nt, data.F, 1 + self.S, 1 + self.S)
        pyro.param(
            "z_trans",
            lambda: z_trans.to(device),
            constraint=constraints.simplex,
        )
        #  pyro.param(
        #      "m_probs",
        #      lambda: torch.full(
        #          (1 + self.S, self.K, self.data.Nt, self.data.F),
        #          0.5,
        #          device=device,
        #      ),
        #      constraint=constraints.unit_interval,
        #  )
        #
        #  self._init_parameters()

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
    def _contraction_identity(logits: torch.Tensor) -> torch.Tensor:
        batch_shape = logits.shape[:-2]
        state_dim = logits.size(-1)
        result = torch.eye(state_dim).log()
        result = result.reshape((1,) * len(batch_shape) + (state_dim, state_dim))
        result = result.repeat(batch_shape + (1, 1))
        return result

    @lazy_property
    def compute_probs(self) -> torch.Tensor:
        theta_probs = torch.zeros(self.K, self.data.Nt, self.data.F)
        nbatch_size = self.nbatch_size
        N = sum(self.data.is_ontarget)
        for ndx in torch.split(torch.arange(N), nbatch_size):
            self.n = ndx
            self.nbatch_size = len(ndx)
            with torch.no_grad(), pyro.plate(
                "particles", size=5, dim=-3
            ), handlers.enum(first_available_dim=-4):
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
                    logp[fsx] += model_tr.nodes[f"{name}_{fsx}"]["funsor"]["log_prob"]
                if fsx == "0":
                    # substitute MAP values of z into p(z=z_map, theta, phi)
                    z_map = funsor.Tensor(self.z_map[ndx, 0].long(), dtype=2)["aois"]
                    logp[fsx] = logp[fsx](**{f"z_{fsx}": z_map})
                    # compute log_measure q for given z_map
                    log_measure = guide_tr.nodes[f"m_0_{fsx}"]["funsor"]["log_measure"]
                    +guide_tr.nodes[f"m_1_{fsx}"]["funsor"]["log_measure"]
                    log_measure = log_measure(**{f"z_{fsx}": z_map})
                else:
                    # substitute MAP values of z into p(z=z_map, theta, phi)
                    z_map = funsor.Tensor(self.z_map[ndx, 1:].long(), dtype=2)[
                        "aois", "frames"
                    ]
                    z_map_prev = funsor.Tensor(self.z_map[ndx, :-1].long(), dtype=2)[
                        "aois", "frames"
                    ]
                    fsx_prev = f"slice(0, {self.data.F-1}, None)"
                    logp[fsx] = logp[fsx](
                        **{f"z_{fsx}": z_map, f"z_{fsx_prev}": z_map_prev}
                    )
                    # compute log_measure q for given z_map
                    log_measure = guide_tr.nodes[f"m_0_{fsx}"]["funsor"]["log_measure"]
                    +guide_tr.nodes[f"m_1_{fsx}"]["funsor"]["log_measure"]
                    log_measure = log_measure(
                        **{f"z_{fsx}": z_map, f"z_{fsx_prev}": z_map_prev}
                    )
                # compute p(z_map, theta | phi) = p(z_map, theta, phi) - p(z_map, phi)
                logp[fsx] = logp[fsx] - logp[fsx].reduce(
                    funsor.ops.logaddexp, f"theta_{fsx}"
                )
                # average over m in p * q
                result[fsx] = (logp[fsx] + log_measure).reduce(
                    funsor.ops.logaddexp, frozenset({f"m_0_{fsx}", f"m_1_{fsx}"})
                )
                # average over particles
                result[fsx] = result[fsx].exp().reduce(funsor.ops.mean, "particles")
            theta_probs[:, ndx, 0] = result["0"].data[:, 1:].permute(1, 0)
            theta_probs[:, ndx, 1:] = (
                result[f"slice(1, {self.data.F}, None)"].data[..., 1:].permute(2, 0, 1)
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
        return result[..., 0, :].exp()

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
        return self.z_probs[..., 1:].sum(-1)

    @property
    def m_probs(self) -> torch.Tensor:
        r"""
        Posterior spot presence probability :math:`q(m=1, z=z_\mathsf{MAP})`.
        """
        return Vindex(torch.permute(pyro.param("m_probs").data, (1, 2, 3, 0)))[
            ..., self.z_map.long()
        ]

    @property
    def z_map(self) -> torch.Tensor:
        return self.z_probs.argmax(-1)
