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
            dist.Poisson(self.lamda).log_prob(torch.arange(self.K).float()).exp()
        )
        result[0, -1] = 1 - result[0, : self.K].sum()
        result[1, : self.K - 1] = (
            dist.Poisson(self.lamda).log_prob(torch.arange(self.K - 1).float()).exp()
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
        result[0] = self.init[0]
        for s in range(self.S):
            for k in range(self.K):
                result[self.K * s + k + 1] = self.init[s + 1] / self.K
        return result

    @property
    def trans_theta(self):
        result = torch.zeros(
            self.K * self.S + 1, self.K * self.S + 1, dtype=torch.float
        )
        for i in range(self.K * self.S + 1):
            # FIXME
            j = (i + 1) // self.K
            result[i, 0] = self.trans[j, 0]
            for s in range(self.S):
                for k in range(self.K):
                    result[i, self.K * s + k + 1] = self.trans[j, s + 1] / self.K
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

    def model(self):

        self.gain = pyro.sample("gain", dist.HalfNormal(50))
        self.init = pyro.sample(
            "init", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1))
        )
        self.trans = pyro.sample(
            "trans",
            dist.Dirichlet(torch.ones(self.S + 1, self.S + 1) / (self.S + 1)).to_event(
                1
            ),
        )
        self.lamda = pyro.sample("lamda", dist.Exponential(1))
        self.proximity = pyro.sample("proximity", dist.Exponential(1)).squeeze()
        self.size = torch.stack(
            (
                torch.tensor(2.0),
                (((self.data.D + 1) / (2 * self.proximity)) ** 2 - 1),
            ),
            dim=-1,
        )

        # test data
        self.spot_model(self.data, self.data_loc, prefix="d")

        # control data
        if self.control:
            self.spot_model(self.control, self.control_loc, prefix="c")

    def guide(self):
        # initialize guide parameters
        self.guide_parameters()

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
            dist.Gamma(
                pyro.param("proximity_loc") * pyro.param("proximity_beta"),
                pyro.param("proximity_beta"),
            ),
        )

        # test data
        self.spot_guide(self.data, prefix="d")

        # control data
        if self.control:
            self.spot_guide(self.control, prefix="c")

    def spot_model(self, data, data_loc, prefix):
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
            background_mean = pyro.sample(
                f"{prefix}/background_mean", dist.HalfNormal(1000)
            )
            background_std = pyro.sample(
                f"{prefix}/background_std", dist.HalfNormal(100)
            )
            ndx = ndx[..., None]
            theta_prev = None
            for fdx in frames:
                # sample background intensity
                background = pyro.sample(
                    f"{prefix}/background_{fdx}",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
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
                            dist.HalfNormal(10000),
                        )
                        width = pyro.sample(
                            f"{prefix}/width_{kdx}_{fdx}",
                            AffineBeta(
                                1.5,
                                2,
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
                        locs / self.gain,
                        1 / self.gain,
                        self.data.offset_samples,
                        self.data.offset_logits,
                    ).to_event(2),
                    obs=Vindex(data[:])[ndx, fdx],
                )
                theta_prev = theta_curr

    def spot_guide(self, data, prefix):
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
            if prefix == "d":
                self.batch_idx = ndx.cpu()

            pyro.sample(
                f"{prefix}/background_mean",
                dist.Delta(pyro.param(f"{prefix}/background_mean_loc")[ndx]),
            )
            pyro.sample(
                f"{prefix}/background_std",
                dist.Delta(pyro.param(f"{prefix}/background_std_loc")[ndx]),
            )
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

    def guide_parameters(self):
        pyro.param(
            "proximity_loc",
            lambda: torch.tensor(0.5),
            constraint=constraints.positive,
        )
        pyro.param(
            "proximity_beta",
            lambda: torch.tensor(100),
            constraint=constraints.positive,
        )
        pyro.param("gain_loc", lambda: torch.tensor(5), constraint=constraints.positive)
        pyro.param(
            "gain_beta", lambda: torch.tensor(100), constraint=constraints.positive
        )
        pyro.param(
            "init_mean", lambda: torch.ones(self.S + 1), constraint=constraints.simplex
        )
        pyro.param(
            "init_size", lambda: torch.tensor(2), constraint=constraints.positive
        )
        pyro.param(
            "trans_mean",
            lambda: torch.ones(self.S + 1, self.S + 1),
            constraint=constraints.simplex,
        )
        pyro.param(
            "trans_size",
            lambda: torch.full((self.S + 1, 1), 2),
            constraint=constraints.positive,
        )
        pyro.param(
            "pi_mean", lambda: torch.ones(self.S + 1), constraint=constraints.simplex
        )
        pyro.param("pi_size", lambda: torch.tensor(2), constraint=constraints.positive)
        pyro.param(
            "lamda_loc", lambda: torch.tensor(0.5), constraint=constraints.positive
        )
        pyro.param(
            "lamda_beta", lambda: torch.tensor(100), constraint=constraints.positive
        )

        pyro.param(
            "d/background_mean_loc",
            lambda: torch.full(
                (self.data.N, 1), self.data.data_median - self.data.offset_median
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "d/background_std_loc",
            lambda: torch.ones(self.data.N, 1),
            constraint=constraints.positive,
        )

        if self.control:
            pyro.param(
                "c/background_mean_loc",
                lambda: torch.full(
                    (self.control.N, 1), self.data.data_median - self.data.offset_median
                ),
                constraint=constraints.positive,
            )
            pyro.param(
                "c/background_std_loc",
                lambda: torch.ones(self.control.N, 1),
                constraint=constraints.positive,
            )

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
