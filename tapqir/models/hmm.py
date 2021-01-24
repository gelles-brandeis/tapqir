import torch
import torch.distributions.constraints as constraints
from pyro.distributions.hmm import _logmatmulexp
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, pyro
from torch.distributions.utils import lazy_property

from tapqir.distributions import AffineBeta, ConvolutedGamma
from tapqir.models.model import Model


class HMM(Model):
    r"""
    for :math:`n=1` to :math:`N`:

        for :math:`n=1` to :math:`F`:

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`
    """
    name = "hmm"

    def __init__(self, S=1, K=2, vectorized=True):
        self.vectorized = vectorized
        super().__init__(S, K)
        self.classify = True

    @lazy_property
    def num_states(self):
        r"""
        Total number of states for the image model given by:

            :math:`2 (1+SK) K`
        """
        return 2 * (1 + self.K * self.S) * self.K

    @property
    def probs_j(self):
        result = torch.zeros(2, self.K + 1, dtype=torch.float)
        result[0, : self.K] = (
            dist.Poisson(pyro.param("rate_j"))
            .log_prob(torch.arange(self.K).float())
            .exp()
        )
        result[0, -1] = 1 - result[0, : self.K].sum()
        result[1, : self.K - 1] = (
            dist.Poisson(pyro.param("rate_j"))
            .log_prob(torch.arange(self.K - 1).float())
            .exp()
        )
        result[1, -2] = 1 - result[0, : self.K - 1].sum()
        return result

    @property
    def probs_m(self):
        # this only works for K=2
        result = torch.zeros(1 + self.K * self.S, self.K, 2, dtype=torch.float)
        result[0, :, 0] = self.probs_j[0, 0] + self.probs_j[0, 1] / 2
        result[0, :, 1] = self.probs_j[0, 2] + self.probs_j[0, 1] / 2
        result[1, 0, 1] = 1
        result[1, 1, 0] = self.probs_j[1, 0]
        result[1, 1, 1] = self.probs_j[1, 1]
        result[2, 0, 0] = self.probs_j[1, 0]
        result[2, 0, 1] = self.probs_j[1, 1]
        result[2, 1, 1] = 1
        return result

    @property
    def init_theta(self):
        result = torch.zeros(self.K * self.S + 1, dtype=torch.float)
        result[0] = pyro.param("init_z")[0]
        for s in range(self.S):
            for k in range(self.K):
                result[self.K * s + k + 1] = pyro.param("init_z")[s + 1] / self.K
        return result

    @property
    def trans_theta(self):
        result = torch.zeros(
            self.K * self.S + 1, self.K * self.S + 1, dtype=torch.float
        )
        for i in range(self.K * self.S + 1):
            # FIXME
            j = (i + 1) // self.K
            result[i, 0] = pyro.param("trans_z")[j, 0]
            for s in range(self.S):
                for k in range(self.K):
                    result[i, self.K * s + k + 1] = (
                        pyro.param("trans_z")[j, s + 1] / self.K
                    )
        return result

    @lazy_property
    def theta_to_z(self):
        result = torch.zeros(self.K * self.S + 1, self.K, dtype=torch.long)
        for s in range(self.S):
            result[1 + s * self.K : 1 + (s + 1) * self.K] = torch.eye(self.K) * (s + 1)
        return result

    @lazy_property
    def ontarget(self):
        return torch.clamp(self.theta_to_z, min=0, max=1)

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
    def j_probs(self):
        r"""
        Probability of an off-target spot :math:`p(j_{knf})`.
        """
        return self.m_probs - self.z_probs

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "sknf,nfs->knf", pyro.param("d/m_probs").data[..., 1], self.theta_probs
        )

    @property
    def z_marginal(self):
        return self.z_probs.sum(-3)

    @property
    def z_map(self):
        return self.z_marginal > 0.5

    @property
    def inference_config_model(self):
        if self.classify:
            return {"hide_types": ["param"]}
        return {"hide": ["width_mean", "width_size", "height_scale"]}

    @property
    def inference_config_guide(self):
        if self.classify:
            return {
                "expose": ["d/theta_probs", "d/m_probs"],
                "expose_types": ["sample"],
            }
        return {"expose_types": ["sample", "param"]}

    @handlers.block(**{"hide": ["width_mean", "width_size", "height_scale"]})
    def model(self):
        # initialize model parameters
        self.model_parameters()

        # test data
        self.spot_model(self.data, self.data_loc, prefix="d")

        # control data
        if self.control:
            self.spot_model(self.control, self.control_loc, prefix="c")

    def guide(self):
        # initialize guide parameters
        self.guide_parameters()

        # test data
        self.spot_guide(self.data, prefix="d")

        # control data
        if self.control:
            self.spot_guide(self.control, prefix="c")

    def spot_model(self, data, data_loc, prefix):
        # spots
        spots = pyro.plate(f"{prefix}/spots", self.K)
        # target sites
        targets = pyro.plate(f"{prefix}/targets", data.N, dim=-2)
        # time frames
        frames = (
            pyro.vectorized_markov(name=f"{prefix}/frames", size=data.F, dim=-1)
            if self.vectorized
            else pyro.markov(range(data.F))
        )

        with targets as ndx:
            ndx = ndx[..., None]
            theta_prev = None
            for fdx in frames:
                # sample background intensity
                background = pyro.sample(
                    f"{prefix}/background_{fdx}",
                    dist.Gamma(
                        Vindex(pyro.param(f"{prefix}/background_loc"))[ndx, 0]
                        * Vindex(pyro.param(f"{prefix}/background_beta"))[ndx, 0],
                        Vindex(pyro.param(f"{prefix}/background_beta"))[ndx, 0],
                    ),
                )
                locs = background[..., None, None]

                # sample hidden model state (1+K*S,)
                if data.dtype == "test":
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
                            dist.HalfNormal(pyro.param("height_scale")),
                        )
                        width = pyro.sample(
                            f"{prefix}/width_{kdx}_{fdx}",
                            AffineBeta(
                                pyro.param("width_mean"),
                                pyro.param("width_size"),
                                0.75,
                                2.25,
                            ),
                        )
                        x = pyro.sample(
                            f"{prefix}/x_{kdx}_{fdx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.D + 1) / 2,
                                (data.D + 1) / 2,
                            ),
                        )
                        y = pyro.sample(
                            f"{prefix}/y_{kdx}_{fdx}",
                            AffineBeta(
                                0,
                                self.size[ontarget],
                                -(data.D + 1) / 2,
                                (data.D + 1) / 2,
                            ),
                        )

                    # calculate image shape w/o offset
                    height = height.masked_fill(m == 0, 0.0)
                    gaussian = data_loc(height, width, x, y, ndx, fdx)
                    locs = locs + gaussian

                # observed data
                pyro.sample(
                    f"{prefix}/data_{fdx}",
                    ConvolutedGamma(
                        locs / pyro.param("gain"),
                        1 / pyro.param("gain"),
                        self.data.offset_samples,
                        self.data.offset_logits,
                    ).to_event(2),
                    obs=Vindex(data[:])[ndx, fdx],
                )
                theta_prev = theta_curr

    def spot_guide(self, data, prefix):
        # spots
        spots = pyro.plate(f"{prefix}/spots", self.K)
        # target sites
        targets = pyro.plate(
            f"{prefix}/targets",
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

        with targets as ndx:
            if prefix == "d":
                self.batch_idx = ndx.cpu()
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
                if data.dtype == "test":
                    probs = (
                        Vindex(pyro.param(f"{prefix}/theta_trans"))[ndx, fdx, 0]
                        if isinstance(fdx, int) and fdx < 1
                        else Vindex(pyro.param(f"{prefix}/theta_trans"))[
                            ndx, fdx, theta_prev
                        ]
                    )
                    theta_curr = pyro.sample(
                        f"{prefix}/theta_{fdx}",
                        dist.Categorical(probs),
                        infer={"enumerate": "parallel"},
                    )
                else:
                    theta_curr = 0

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
                                -(data.D + 1) / 2,
                                (data.D + 1) / 2,
                            ),
                        )
                        pyro.sample(
                            f"{prefix}/y_{kdx}_{fdx}",
                            AffineBeta(
                                Vindex(pyro.param(f"{prefix}/y_mean"))[kdx, ndx, fdx],
                                Vindex(pyro.param(f"{prefix}/size"))[kdx, ndx, fdx],
                                -(data.D + 1) / 2,
                                (data.D + 1) / 2,
                            ),
                        )
                theta_prev = theta_curr

    def model_parameters(self):
        pyro.param(
            "proximity",
            lambda: torch.tensor([0.5]),
            constraint=constraints.interval(0.01, 2.0),
        )
        self.size = torch.cat(
            (
                torch.tensor([2.0]),
                (((self.data.D + 1) / (2 * pyro.param("proximity"))) ** 2 - 1),
            ),
            dim=-1,
        )
        pyro.param("gain", lambda: torch.tensor(5.0), constraint=constraints.positive)
        pyro.param(
            "init_z", lambda: torch.ones(self.S + 1), constraint=constraints.simplex
        )
        pyro.param(
            "trans_z",
            lambda: torch.ones(self.S + 1, self.S + 1),
            constraint=constraints.simplex,
        )
        pyro.param("rate_j", lambda: torch.tensor(0.5), constraint=constraints.positive)

        pyro.param(
            "d/background_loc",
            lambda: torch.full(
                (self.data.N, 1), self.data.data_median - self.data.offset_median
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "d/background_beta",
            lambda: torch.ones(self.data.N, 1),
            constraint=constraints.positive,
        )

        if self.control:
            pyro.param(
                "c/background_loc",
                lambda: torch.full(
                    (self.control.N, 1), self.data.data_median - self.data.offset_median
                ),
                constraint=constraints.positive,
            )
            pyro.param(
                "c/background_beta",
                lambda: torch.ones(self.control.N, 1),
                constraint=constraints.positive,
            )

        pyro.param(
            "width_mean",
            lambda: torch.tensor([1.5]),
            constraint=constraints.interval(0.75, 2.25),
        )
        pyro.param(
            "width_size", lambda: torch.tensor([2.0]), constraint=constraints.positive
        )
        pyro.param(
            "height_scale",
            lambda: torch.tensor(10000.0),
            constraint=constraints.positive,
        )

    def guide_parameters(self):
        self.spot_parameters(self.data, prefix="d")

        if self.control:
            self.spot_parameters(self.control, prefix="c")

    def spot_parameters(self, data, prefix):
        pyro.param(
            f"{prefix}/theta_trans",
            lambda: torch.ones(
                data.N, data.F, 1 + self.K * self.S, 1 + self.K * self.S
            ),
            constraint=constraints.simplex,
        )
        m_probs = torch.ones(1 + self.K * self.S, self.K, data.N, data.F, 2)
        m_probs[1, 0, :, :, 0] = 0
        m_probs[2, 1, :, :, 0] = 0
        pyro.param(f"{prefix}/m_probs", lambda: m_probs, constraint=constraints.simplex)
        pyro.param(
            f"{prefix}/b_loc",
            lambda: torch.full(
                (data.N, data.F), self.data.data_median - self.data.offset_median
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/b_beta",
            lambda: torch.ones(data.N, data.F),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_loc",
            lambda: torch.full((self.K, data.N, data.F), 2000.0),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_beta",
            lambda: torch.full((self.K, data.N, data.F), 0.001),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/w_mean",
            lambda: torch.full((self.K, data.N, data.F), 1.5),
            constraint=constraints.interval(0.75, 2.25),
        )
        pyro.param(
            f"{prefix}/w_size",
            lambda: torch.full((self.K, data.N, data.F), 100.0),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            f"{prefix}/x_mean",
            lambda: torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(-(data.D + 1) / 2, (data.D + 1) / 2),
        )
        pyro.param(
            f"{prefix}/y_mean",
            lambda: torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(-(data.D + 1) / 2, (data.D + 1) / 2),
        )
        size = torch.ones(self.K, data.N, data.F) * 200.0
        if self.K == 2:
            size[1] = 7.0
        elif self.K == 3:
            size[1] = 7.0
            size[2] = 3.0
        pyro.param(
            f"{prefix}/size", lambda: size, constraint=constraints.greater_than(2.0)
        )
