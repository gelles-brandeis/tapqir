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
from cosmos.models.utils import pi_m_calc, pi_theta_calc
from cosmos.models.utils import z_probs_calc


class Tracker(Model):
    """ Track on-target Spot """

    def __init__(self, S):
        self.__name__ = "tracker"
        super().__init__(S)

    @poutine.block(hide=["width_mode", "width_size"])
    def model(self):
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.S+1)))
        lamda = pyro.sample("lamda", dist.Dirichlet(0.5 * torch.ones(self.S+1)))
        pi_m = pi_m_calc(lamda, self.S)
        pi_theta = pi_theta_calc(pi, self.K, self.S)

        with scope(prefix="d"):
            self.spot_model(self.data, self.data_loc, pi_m, pi_theta, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_model(self.control, self.control_loc, pi_m, None, prefix="c")

    @config_enumerate
    def guide(self):
        pyro.sample("pi", dist.Dirichlet(param("pi_mode") * param("pi_size")))
        pyro.sample("lamda", dist.Dirichlet(param("lamda_mode") * param("lamda_size")))

        with scope(prefix="d"):
            self.spot_guide(self.data, True, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_guide(self.control, False, prefix="c")

    def spot_model(self, data, data_loc, pi_m, pi_theta, prefix):
        K_plate = pyro.plate("K_plate", self.K)
        N_plate = pyro.plate("N_plate", data.N, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, dim=-1)

        with N_plate as batch_idx, F_plate as frame_idx:
            batch_idx = batch_idx[:, None]
            background = pyro.sample(
                "background", dist.Gamma(
                    Vindex(param(f"{prefix}/background_loc"))[batch_idx]
                    * param("background_beta"), param("background_beta")))
            locs = background[..., None, None]

            if pi_theta is not None:
                theta = pyro.sample("theta", dist.Categorical(pi_theta))
            else:
                theta = 0

            for k_idx in K_plate:
                theta_mask = Vindex(self.theta_matrix)[theta, k_idx]
                m_mask = pyro.sample(f"m_{k_idx}", dist.Categorical(Vindex(pi_m)[theta_mask]))
                with pyro.poutine.mask(mask=m_mask > 0):
                    height = pyro.sample(
                        f"height_{k_idx}", dist.HalfNormal(10000.)
                    )
                    width = pyro.sample(
                        f"width_{k_idx}", AffineBeta(
                            param("width_mode"),
                            param("width_size"), 0.75, 1.25))
                    x = pyro.sample(
                        f"x_{k_idx}", AffineBeta(
                            0, self.size[theta_mask], -(data.D+1)/2, data.D+1))
                    y = pyro.sample(
                        f"y_{k_idx}", AffineBeta(
                            0, self.size[theta_mask], -(data.D+1)/2, data.D+1))

                    height = height.masked_fill(m_mask == 0, 0.)

                    gaussian = data_loc(height, width, x, y, batch_idx, frame_idx)
                    locs = locs + gaussian

            pyro.sample(
                "data", ConvolutedGamma(
                    locs / param("gain"), 1 / param("gain"),
                    self.offset_samples, self.offset_weights.log()
                ).to_event(2),
                obs=Vindex(data.data)[batch_idx, frame_idx, :, :]
            )

    def spot_guide(self, data, theta, prefix):
        K_plate = pyro.plate("K_plate", self.K)
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.batch_size, subsample=self.n, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, subsample=self.frames, dim=-1)

        with N_plate as batch_idx, F_plate as frame_idx:
            batch_idx = batch_idx[:, None]
            pyro.sample(
                "background", dist.Gamma(
                    Vindex(param(f"{prefix}/b_loc"))[batch_idx, frame_idx]
                    * Vindex(param(f"{prefix}/b_beta"))[batch_idx, frame_idx],
                    Vindex(param(f"{prefix}/b_beta"))[batch_idx, frame_idx]))

            if theta:
                theta = pyro.sample("theta", dist.Categorical(
                    Vindex(param(f"{prefix}/theta_probs"))[batch_idx, frame_idx, :]))
            else:
                theta = 0

            for k_idx in K_plate:
                theta_mask = Vindex(self.theta_matrix)[theta, k_idx]
                m_mask = pyro.sample(f"m_{k_idx}", dist.Categorical(
                    Vindex(param(f"{prefix}/m_probs"))[k_idx, batch_idx, frame_idx, theta_mask, :]))
                with pyro.poutine.mask(mask=m_mask > 0):
                    pyro.sample(
                        f"height_{k_idx}",
                        dist.Gamma(
                            Vindex(param(f"{prefix}/h_loc"))[k_idx, batch_idx, frame_idx]
                            * Vindex(param(f"{prefix}/h_beta"))[k_idx, batch_idx, frame_idx],
                            Vindex(param(f"{prefix}/h_beta"))[k_idx, batch_idx, frame_idx]
                        )
                    )
                    pyro.sample(
                        f"width_{k_idx}", AffineBeta(
                            Vindex(param(f"{prefix}/w_mode"))[k_idx, batch_idx, frame_idx],
                            Vindex(param(f"{prefix}/w_size"))[k_idx, batch_idx, frame_idx],
                            0.75, 1.25))
                    pyro.sample(
                        f"x_{k_idx}", AffineBeta(
                            Vindex(param(f"{prefix}/x_mode"))[k_idx, batch_idx, frame_idx],
                            Vindex(param(f"{prefix}/size"))[k_idx, batch_idx, frame_idx],
                            -(data.D+1)/2, data.D+1))
                    pyro.sample(
                        f"y_{k_idx}", AffineBeta(
                            Vindex(param(f"{prefix}/y_mode"))[k_idx, batch_idx, frame_idx],
                            Vindex(param(f"{prefix}/size"))[k_idx, batch_idx, frame_idx],
                            -(data.D+1)/2, data.D+1))

    def guide_parameters(self):
        self.spot_parameters(self.data, True, True, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, True, False, prefix="c")

    def spot_parameters(self, data, m, theta, prefix):
        param(f"{prefix}/background_loc",
              torch.ones(data.N) * (self.data_median - self.offset_median),
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
              torch.ones(self.K, data.N, data.F) * 1.3,
              constraint=constraints.interval(0.75, 2.))
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

        if m:
            m_probs = torch.zeros(self.K, data.N, data.F, self.S+1, self.S+1)
            m_probs[..., 0, :] = 1
            for s in range(1, self.S+1):
                m_probs[..., s, s] = 1
            param(f"{prefix}/m_probs",
                  m_probs,
                  constraint=constraints.simplex)
        if theta:
            theta_probs = torch.ones(
                data.N, data.F, self.S*self.K+1)
            param(f"{prefix}/theta_probs", theta_probs,
                  constraint=constraints.simplex)

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+1)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        param("gain", torch.tensor(5.), constraint=constraints.positive)
        param("pi_mode", torch.ones(self.S+1), constraint=constraints.simplex)
        param("lamda_mode", torch.ones(self.S+1), constraint=constraints.simplex)
        param("pi_size", torch.tensor([1000.]), constraint=constraints.positive)
        param("lamda_size", torch.tensor([1000.]), constraint=constraints.positive)
        param("background_beta", torch.tensor([1.]),
              constraint=constraints.positive)
        param("width_mode", torch.tensor([1.3]),
              constraint=constraints.interval(0.75, 2.))
        param("width_size",
              torch.tensor([2.]), constraint=constraints.positive)

    def infer(self):
        z_probs = z_probs_calc(pyro.param("d/theta_probs"))
        # k_probs = k_probs_calc(pyro.param("d/m_probs"), pyro.param("d/theta_probs"))
        self.predictions["z_prob"] = z_probs.squeeze()
        self.predictions["z"] = \
            self.predictions["z_prob"] > 0.5
        # self.predictions["m_prob"] = k_probs.squeeze()
        # self.predictions["m"] = self.predictions["m_prob"] > 0.5
        np.save(os.path.join(self.path, "predictions.npy"),
                self.predictions)
