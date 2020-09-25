import torch
import pyro
import numpy as np
import os
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro import param
from pyro import poutine
from pyro.contrib.autoname import scope
from pyro.ops.indexing import Vindex
import torch.distributions.constraints as constraints

from cosmos.models import Model
from cosmos.distributions import AffineBeta, ConvolutedGamma


class Tracker(Model):
    r"""
    for :math:`n=1` to :math:`N`:

        for :math:`n=1` to :math:`F`:

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`
    """
    name = "kplate"

    def __init__(self, S):
        super().__init__(S)

    @poutine.block(hide=["width_mode", "width_size", "proximity",
                         "offset_samples", "offset_weights"])
    def model(self):
        self.model_parameters()
        self.size = torch.cat((torch.tensor([2.]), (((self.data.D+1) / (2 * param("proximity"))) ** 2 - 1)), dim=-1)

        with scope(prefix="d"):
            self.spot_model(self.data, self.data_loc, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_model(self.control, self.control_loc, prefix="c")

    @config_enumerate
    def guide(self):
        self.guide_parameters()
        with scope(prefix="d"):
            self.spot_guide(self.data, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_guide(self.control, prefix="c")

    def spot_model(self, data, data_loc, prefix):
        # conditionally independent plates
        K_plate = pyro.plate("K_plate", self.K, dim=-3)
        N_plate = pyro.plate("N_plate", data.N, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, dim=-1)

        with N_plate as ndx, F_plate as fdx:
            ndx = ndx[:, None]
            # sample background intensity
            background = pyro.sample(
                "background", dist.Gamma(
                    Vindex(param(f"{prefix}/background_loc"))[ndx]
                    * Vindex(param(f"{prefix}/background_beta"))[ndx],
                    Vindex(param(f"{prefix}/background_beta"))[ndx]))

            state = pyro.sample("state", dist.Categorical(logits=self.logits_state))

            with K_plate as kdx:
                kdx = kdx[:, None, None]
                m_mask = Vindex(self.state_to_m)[state, kdx].bool()
                ontarget = Vindex(self.ontarget)[state, kdx]

                with pyro.poutine.mask(mask=m_mask):
                    height = pyro.sample(
                        "height", dist.HalfNormal(10000.)
                    )
                    width = pyro.sample(
                        "width", AffineBeta(
                            param("width_mode"),
                            param("width_size"), 0.75, 2.25))
                    x = pyro.sample(
                        "x", AffineBeta(
                            0, self.size[ontarget], -(data.D+1)/2, (data.D+1)/2))
                    y = pyro.sample(
                        "y", AffineBeta(
                            0, self.size[ontarget], -(data.D+1)/2, (data.D+1)/2))

            height = height.masked_fill(~m_mask, 0.)

            locs = background[..., None, None] + data_loc(height, width, x, y, ndx, fdx).sum(-5, keepdim=True)

            pyro.sample(
                "data", ConvolutedGamma(
                    locs / param("gain"), 1 / param("gain"),
                    param("offset_samples"), param("offset_weights").log()
                ).to_event(2),
                obs=Vindex(data.data)[ndx, fdx, :, :]
            )

    def spot_guide(self, data, prefix):
        K_plate = pyro.plate("K_plate", self.K, dim=-3)
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.batch_size, subsample=self.n, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, subsample=self.frames, dim=-1)

        with N_plate as ndx, F_plate as fdx:
            ndx = ndx[:, None]
            pyro.sample(
                "background", dist.Gamma(
                    Vindex(param(f"{prefix}/b_loc"))[ndx, fdx]
                    * Vindex(param(f"{prefix}/b_beta"))[ndx, fdx],
                    Vindex(param(f"{prefix}/b_beta"))[ndx, fdx]))

            state = pyro.sample("state", dist.Categorical(
                    logits=Vindex(param(f"{prefix}/logits_state"))[ndx, fdx]))

            with K_plate as kdx:
                kdx = kdx[:, None, None]
                m_mask = Vindex(self.state_to_m)[state, kdx].bool()

                with pyro.poutine.mask(mask=m_mask):
                    pyro.sample(
                        "height",
                        dist.Gamma(
                            Vindex(param(f"{prefix}/h_loc"))[kdx, ndx, fdx]
                            * Vindex(param(f"{prefix}/h_beta"))[kdx, ndx, fdx],
                            Vindex(param(f"{prefix}/h_beta"))[kdx, ndx, fdx]
                        )
                    )
                    pyro.sample(
                        "width", AffineBeta(
                            Vindex(param(f"{prefix}/w_mode"))[kdx, ndx, fdx],
                            Vindex(param(f"{prefix}/w_size"))[kdx, ndx, fdx],
                            0.75, 2.25))
                    pyro.sample(
                        "x", AffineBeta(
                            Vindex(param(f"{prefix}/x_mode"))[kdx, ndx, fdx],
                            Vindex(param(f"{prefix}/size"))[kdx, ndx, fdx],
                            -(data.D+1)/2, (data.D+1)/2))
                    pyro.sample(
                        "y", AffineBeta(
                            Vindex(param(f"{prefix}/y_mode"))[kdx, ndx, fdx],
                            Vindex(param(f"{prefix}/size"))[kdx, ndx, fdx],
                            -(data.D+1)/2, (data.D+1)/2))

    def guide_parameters(self):
        self.spot_parameters(self.data, True, True, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, True, False, prefix="c")

    def spot_parameters(self, data, m, theta, prefix):
        param(f"{prefix}/background_loc",
              torch.ones(data.N) * (self.data_median - self.offset_median),
              constraint=constraints.positive)
        param(f"{prefix}/background_beta", torch.ones(data.N),
              constraint=constraints.positive)
        param(f"{prefix}/b_loc",
              (self.data_median - self.offset_median).repeat(data.N, data.F),
              constraint=constraints.positive)
        param(f"{prefix}/b_beta",
              torch.ones(data.N, data.F) * 30,
              constraint=constraints.positive)
        param(f"{prefix}/h_loc",
              (self.noise * 2).repeat(self.K, data.N, data.F),
              constraint=constraints.positive)
        param(f"{prefix}/h_beta",
              torch.ones(self.K, data.N, data.F),
              constraint=constraints.positive)
        param(f"{prefix}/w_mode",
              torch.ones(self.K, data.N, data.F) * 1.5,
              constraint=constraints.interval(0.75, 2.25))
        param(f"{prefix}/w_size",
              torch.ones(self.K, data.N, data.F) * 100.,
              constraint=constraints.greater_than(2.))
        param(f"{prefix}/x_mode",
              torch.zeros(self.K, data.N, data.F),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        param(f"{prefix}/y_mode",
              torch.zeros(self.K, data.N, data.F),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        size = torch.ones(self.K, data.N, data.F) * 5.
        size[0] = ((data.D+1) / (2*0.5)) ** 2 - 1
        param(f"{prefix}/size",
              size, constraint=constraints.greater_than(2.))

        param(f"{prefix}/logits_state",
              torch.ones(data.N, data.F, self.num_states),
              constraint=constraints.real)

    def model_parameters(self):
        # Global Parameters
        param("proximity", torch.tensor([0.5]),
              constraint=constraints.interval(0.01, 2.))
        param("gain", torch.tensor(5.), constraint=constraints.positive)
        param("logits_z", (torch.ones(self.S+1)/(self.S+1)).log(), constraint=constraints.real)
        param("rate_j", torch.tensor(0.5), constraint=constraints.positive)
        param("width_mode", torch.tensor([1.5]),
              constraint=constraints.interval(0.75, 2.25))
        param("width_size",
              torch.tensor([2.]), constraint=constraints.positive)
        param("offset_samples",
              self.offset_samples,
              constraint=constraints.positive)
        param("offset_weights",
              self.offset_weights,
              constraint=constraints.positive)

    def infer(self):
        self.predictions["z_prob"] = self.z_probs[..., 1].sum(-1).cpu().numpy()
        self.predictions["z"] = \
            self.predictions["z_prob"] > 0.5
        np.save(os.path.join(self.path, "predictions.npy"),
                self.predictions)
