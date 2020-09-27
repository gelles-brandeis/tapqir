import torch
import torch.distributions.constraints as constraints
from torch.distributions.utils import probs_to_logits

from pyro import param, sample, plate, poutine
from pyro.distributions import Categorical, Gamma, HalfNormal
from pyro.infer import config_enumerate
from pyro.contrib.autoname import scope

from cosmos.distributions import AffineBeta, ConvolutedGamma
from cosmos.models import Model


class Masked(Model):
    r"""
    for :math:`n=1` to :math:`N`:

        for :math:`n=1` to :math:`F`:

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`
    """
    name = "masked"

    def __init__(self, S):
        super().__init__(S)

    @poutine.block(hide=["width_mode", "width_size", "proximity",
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
                "height", HalfNormal(10000.).mask(m_mask).to_event(1)
            )
            width = sample(
                "width", AffineBeta(
                    param("width_mode"),
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
                "data", ConvolutedGamma(
                    locs / param("gain"), 1 / param("gain"),
                    param("offset_samples"), probs_to_logits(param("offset_weights"))
                ).to_event(2),
                obs=data[ndx]
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
                    param(f"{prefix}/w_mode")[ndx],
                    param(f"{prefix}/w_size")[ndx],
                    0.75, 2.25
                ).mask(m_mask).to_event(1)
            )
            sample(
                "x", AffineBeta(
                    param(f"{prefix}/x_mode")[ndx],
                    param(f"{prefix}/size")[ndx],
                    -(data.D+1)/2, (data.D+1)/2
                ).mask(m_mask).to_event(1))
            sample(
                "y", AffineBeta(
                    param(f"{prefix}/y_mode")[ndx],
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
              torch.ones(self.data.N, 1) * (self.data_median - self.offset_median),
              constraint=constraints.positive)
        param("d/background_beta", torch.ones(self.data.N, 1),
              constraint=constraints.positive)

        if self.control:
            param("c/background_loc",
                  torch.ones(self.control.N, 1) * (self.data_median - self.offset_median),
                  constraint=constraints.positive)
            param("c/background_beta", torch.ones(self.control.N, 1),
                  constraint=constraints.positive)

        param("width_mode",
              torch.tensor([1.5]),
              constraint=constraints.interval(0.75, 2.25))
        param("width_size",
              torch.tensor([2.]),
              constraint=constraints.positive)

        param("offset_samples",
              self.offset_samples,
              constraint=constraints.positive)
        param("offset_weights",
              self.offset_weights,
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
        param(f"{prefix}/w_mode",
              torch.ones(data.N, data.F, self.K) * 1.5,
              constraint=constraints.interval(0.75, 2.25))
        param(f"{prefix}/w_size",
              torch.ones(data.N, data.F, self.K) * 100.,
              constraint=constraints.greater_than(2.))
        param(f"{prefix}/x_mode",
              torch.zeros(data.N, data.F, self.K),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        param(f"{prefix}/y_mode",
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
