import itertools
import torch
import torch.distributions.constraints as constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs

from pyro import param, sample, plate, poutine
from pyro.distributions import Categorical, Gamma, Poisson
from pyro.infer import config_enumerate
from pyro.ops.indexing import Vindex
from pyro.contrib.autoname import scope

from cosmos.distributions import AffineBeta, FixedOffsetGamma
from cosmos.models import Model


class FixedOffset(Model):
    r"""
    for :math:`n=1` to :math:`N`:

        for :math:`n=1` to :math:`F`:

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`
    """
    name = "fixedoffset"

    def __init__(self, S, K):
        super().__init__(S, K)

    @property
    def num_states(self):
        r"""
        Total number of states for the image model given by:

            :math:`2^K + S K 2^{K-1}`
        """
        return 2**self.K + self.S * self.K * 2**(self.K-1)

    @property
    def logits_j(self):
        result = torch.zeros(2, self.K+1, dtype=torch.float)
        result[0, :self.K] = Poisson(param("rate_j")).log_prob(torch.arange(self.K).float())
        result[0, -1] = torch.log1p(-result[0, :self.K].exp().sum())
        result[1, :self.K-1] = Poisson(param("rate_j")).log_prob(torch.arange(self.K-1).float())
        result[1, -2] = torch.log1p(-result[0, :self.K-1].exp().sum())
        return result

    @property
    def logits_state(self):
        logits_z = Vindex(param("logits_z"))[self.state_to_z.sum(-1)]
        logits_j = Vindex(self.logits_j)[self.ontarget.sum(-1), self.state_to_j.sum(-1)]
        _, idx, counts = torch.unique(
            torch.stack((self.state_to_z.sum(-1), self.state_to_j.sum(-1)), -1),
            return_counts=True, return_inverse=True, dim=0)
        return logits_z + logits_j - torch.log(Vindex(counts)[idx].float())

    @property
    def state_to_z(self):
        result = torch.zeros(self.num_states, self.K, dtype=torch.long)
        for i in range(2**self.K, self.num_states):
            s, r = divmod(i - 2**self.K, self.K * 2**(self.K-1))
            k, t = divmod(r, 2**(self.K-1))
            result[i, k] = s+1
        return result

    @property
    def ontarget(self):
        return torch.clamp(self.state_to_z, min=0, max=1)

    @property
    def state_to_j(self):
        result = torch.zeros(self.num_states, self.K, dtype=torch.long)
        k_lst = torch.tensor(list(itertools.product([0, 1], repeat=self.K)), dtype=torch.long)
        km1_lst = torch.tensor(list(itertools.product([0, 1], repeat=self.K-1)), dtype=torch.long)
        kdx = torch.arange(self.K)
        result[:2**self.K] = k_lst
        for s in range(self.S):
            for k in range(self.K):
                km1dx = torch.cat([kdx[:k], kdx[k+1:]])
                result[2**self.K+(s*self.K+k)*2**(self.K-1):2**self.K+(s*self.K+k+1)*2**(self.K-1), km1dx] = km1_lst
        return result

    @property
    def state_to_m(self):
        return torch.clamp(self.state_to_z + self.state_to_j, min=0, max=1)

    @property
    def z_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        return torch.einsum(
            "nfi,iks->nfks",
            logits_to_probs(param("d/logits_state").data),
            torch.eye(self.S+1)[self.state_to_z])

    @property
    def j_probs(self):
        r"""
        Probability of an off-target spot :math:`p(j_{knf})`.
        """
        return torch.einsum(
            "nfi,ikt->nfkt",
            logits_to_probs(param("d/logits_state").data),
            torch.eye(2)[self.state_to_j])

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "nfi,ikt->nfkt",
            logits_to_probs(param("d/logits_state").data),
            torch.eye(2)[self.state_to_m])

    @property
    def z_marginal(self):
        return self.z_probs[..., 1:].sum(dim=(-2, -1))

    @poutine.block(hide=["width_mean", "width_size", "proximity",
                         "offset_samples", "offset_weights"])
    def model(self):
        # initialize model parameters
        self.model_parameters()

        # test data
        with scope(prefix="d"):
            self.spot_model(self.data, self.data_loc, prefix="d")

        # control data
        if self.control:
            with scope(prefix="c"):
                self.spot_model(self.control, self.control_loc, prefix="c")

    @config_enumerate
    def guide(self):
        # initialize guide parameters
        self.guide_parameters()

        # test data
        with scope(prefix="d"):
            self.spot_guide(self.data, prefix="d")

        # control data
        if self.control:
            with scope(prefix="c"):
                self.spot_guide(self.control, prefix="c")

    def spot_model(self, data, data_loc, prefix):
        # target sites
        N_plate = plate("N_plate", data.N, dim=-2)
        # time frames
        F_plate = plate("F_plate", data.F, dim=-1)

        with N_plate as ndx, F_plate:
            # sample background intensity
            background = sample(
                "background", Gamma(
                    param(f"{prefix}/background_loc")[ndx]
                    * param(f"{prefix}/background_beta")[ndx],
                    param(f"{prefix}/background_beta")[ndx]
                )
            )

            # sample hidden model state
            state = sample("state", Categorical(logits=self.logits_state))

            m_mask = self.state_to_m[state].bool()
            ontarget = self.ontarget[state]

            # sample spot variables
            height = sample(
                "height", Gamma(param("height_loc") / param("gain"), 1 / param("gain")).mask(m_mask).to_event(1)
            )
            width = sample(
                "width", AffineBeta(
                    param("width_mean"),
                    param("width_size"), 0.75, 2.25
                ).mask(m_mask).to_event(1))
            x = sample(
                "x", AffineBeta(
                    0, self.size[ontarget], -(data.D+1)/2, (data.D+1)/2
                ).mask(m_mask).to_event(1))
            y = sample(
                "y", AffineBeta(
                    0, self.size[ontarget], -(data.D+1)/2, (data.D+1)/2
                ).mask(m_mask).to_event(1))

            # calculate image shape w/o offset
            height = height.masked_fill(~m_mask, 0.)
            locs = background[..., None, None] + data_loc(height, width, x, y, ndx).sum(-3)

            # observed data
            sample(
                "data", FixedOffsetGamma(
                    locs, param("gain"), param("offset")
                ).to_event(2),
            )

    def spot_guide(self, data, prefix):
        # target sites
        N_plate = plate("N_plate", data.N,
                        subsample_size=self.batch_size,
                        subsample=self.n, dim=-2)
        # time frames
        F_plate = plate("F_plate", data.F, dim=-1)

        with N_plate as ndx, F_plate:
            # sample background intensity
            sample(
                "background", Gamma(
                    param(f"{prefix}/b_loc")[ndx]
                    * param(f"{prefix}/b_beta")[ndx],
                    param(f"{prefix}/b_beta")[ndx]))

            # sample hidden model state
            state = sample("state", Categorical(
                    logits=param(f"{prefix}/logits_state")[ndx]))

            m_mask = self.state_to_m[state].bool()

            # sample spot variables
            sample(
                "height", Gamma(
                    param(f"{prefix}/h_loc")[ndx] * param(f"{prefix}/h_beta")[ndx],
                    param(f"{prefix}/h_beta")[ndx]
                ).mask(m_mask).to_event(1)
            )
            sample(
                "width", AffineBeta(
                    param(f"{prefix}/w_mean")[ndx],
                    param(f"{prefix}/w_size")[ndx],
                    0.75, 2.25
                ).mask(m_mask).to_event(1)
            )
            sample(
                "x", AffineBeta(
                    param(f"{prefix}/x_mean")[ndx],
                    param(f"{prefix}/size")[ndx],
                    -(data.D+1)/2, (data.D+1)/2
                ).mask(m_mask).to_event(1))
            sample(
                "y", AffineBeta(
                    param(f"{prefix}/y_mean")[ndx],
                    param(f"{prefix}/size")[ndx],
                    -(data.D+1)/2, (data.D+1)/2
                ).mask(m_mask).to_event(1))

    def model_parameters(self):
        param("proximity",
              torch.tensor([0.5]),
              constraint=constraints.interval(0.01, 2.))
        self.size = torch.cat((
            torch.tensor([2.]),
            (((self.data.D+1) / (2*param("proximity"))) ** 2 - 1)
        ), dim=-1)
        param("gain",
              torch.tensor(10.),
              constraint=constraints.positive)
        param("logits_z",
              probs_to_logits(torch.ones(self.S+1) / (self.S+1)),
              constraint=constraints.real)
        param("rate_j",
              torch.tensor(0.5),
              constraint=constraints.positive)

        param("d/background_loc",
              torch.ones(self.data.N, 1) * 150.,
              constraint=constraints.positive)
        param("d/background_beta", torch.ones(self.data.N, 1),
              constraint=constraints.positive)

        if self.control:
            param("c/background_loc",
                  torch.ones(self.control.N, 1) * 150.,
                  constraint=constraints.positive)
            param("c/background_beta", torch.ones(self.control.N, 1),
                  constraint=constraints.positive)

        param("width_mean",
              torch.tensor([1.5]),
              constraint=constraints.interval(0.75, 2.25))
        param("width_size",
              torch.tensor([2.]),
              constraint=constraints.positive)

        param("offset",
              torch.tensor(90.),
              constraint=constraints.positive)

    def guide_parameters(self):
        self.spot_parameters(self.data, prefix="d")

        if self.control:
            self.spot_parameters(self.control, prefix="c")

    def spot_parameters(self, data, prefix):
        param(f"{prefix}/logits_state",
              torch.ones(data.N, data.F, self.num_states),
              constraint=constraints.real)
        param(f"{prefix}/b_loc",
              (self.data_median - self.offset_median).repeat(data.N, data.F),
              constraint=constraints.positive)
        param(f"{prefix}/b_beta",
              torch.ones(data.N, data.F) * 30,
              constraint=constraints.positive)
        param(f"{prefix}/h_loc",
              (self.noise * 2).repeat(data.N, data.F, self.K),
              constraint=constraints.positive)
        param(f"{prefix}/h_beta",
              torch.ones(data.N, data.F, self.K),
              constraint=constraints.positive)
        param(f"{prefix}/w_mean",
              torch.ones(data.N, data.F, self.K) * 1.5,
              constraint=constraints.interval(0.75, 2.25))
        param(f"{prefix}/w_size",
              torch.ones(data.N, data.F, self.K) * 100.,
              constraint=constraints.greater_than(2.))
        param(f"{prefix}/x_mean",
              torch.zeros(data.N, data.F, self.K),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        param(f"{prefix}/y_mean",
              torch.zeros(data.N, data.F, self.K),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        size = torch.ones(data.N, data.F, self.K) * 200.
        if self.K == 2:
            size[..., 1] = 7.
        elif self.K == 3:
            size[..., 1] = 7.
            size[..., 2] = 3.
        param(f"{prefix}/size",
              size, constraint=constraints.greater_than(2.))

    def infer(self):
        self.predictions["z_prob"] = self.z_probs[..., 1].sum(-1).cpu().numpy()
        self.predictions["z"] = \
            self.predictions["z_prob"] > 0.5
