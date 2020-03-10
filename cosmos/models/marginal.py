import torch
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro import param
from pyro import poutine
from pyro.contrib.autoname import scope
from pyro.ops.indexing import Vindex
from cosmos.models.helper import ScaledBeta
import torch.distributions.constraints as constraints

from cosmos.models.model import Model
from cosmos.models.helper import pi_m_calc, pi_theta_calc


class Marginal(Model):
    """ Track on-target Spot """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "marginal"
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

    @poutine.block(hide=["width_mode", "width_size"])
    @config_enumerate
    def model(self):
        self.model_parameters()
        data_pi_m = pi_m_calc(param("pi"), param("lamda"), self.K)
        control_pi_m = pi_m_calc(torch.tensor([1., 0.]), param("lamda"), self.K)
        pi_theta = pi_theta_calc(param("pi"), param("lamda"), self.K)

        with scope(prefix="d"):
            self.spot_model(self.data, data_pi_m, pi_theta, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_model(self.control, control_pi_m, None, prefix="c")

    def guide(self):
        self.guide_parameters()
        with scope(prefix="d"):
            self.spot_guide(self.data, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_guide(self.control, prefix="c")

    def guide_parameters(self):
        self.spot_parameters(self.data, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, prefix="c")


    def spot_model(self, data, pi_m, pi_theta, prefix):
        N_plate = pyro.plate("N_plate", data.N, dim=-5)
        F_plate = pyro.plate("F_plate", data.F, dim=-4)
        X_plate = pyro.plate("X_plate", data.D, dim=-3)
        Y_plate = pyro.plate("Y_plate", data.D, dim=-2)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)

        with N_plate as batch_idx, F_plate:
            background = pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/background_loc")[batch_idx]
                    * param("background_beta"), param("background_beta")))

            m = pyro.sample("m", dist.Categorical(pi_m))
            if pi_theta is not None:
                theta = pyro.sample("theta", dist.Categorical(pi_theta[m]))
                theta_mask = self.theta_matrix[theta.squeeze(dim=-1)]
            else:
                theta_mask = 0
            m_mask = self.m_matrix[m.squeeze(dim=-1)].bool()

            with pyro.poutine.mask(mask=m_mask), K_plate:
                height = pyro.sample(
                    "height", dist.Gamma(
                        param("height_loc") * param("height_beta"),
                        param("height_beta")))
                width = pyro.sample(
                    "width", ScaledBeta(
                        param("width_mode"),
                        param("width_size"), 0.5, 2.5))
                x0 = pyro.sample(
                    "x0", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+3)/2, data.D+3))
                y0 = pyro.sample(
                    "y0", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+3)/2, data.D+3))

            width = width * 2.5 + 0.5
            x0 = x0 * (data.D+3) - (data.D+3)/2
            y0 = y0 * (data.D+3) - (data.D+3)/2

            locs = data.loc(batch_idx, m_mask,
                            height, width, x0, y0, background)
            with X_plate, Y_plate:
                pyro.sample(
                    "data", self.CameraUnit(
                        locs, param("gain"), param("offset")),
                    obs=data[batch_idx].unsqueeze(dim=-1))

    def spot_guide(self, data, prefix):
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-5)
        F_plate = pyro.plate("F_plate", data.F, dim=-4)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)

        with N_plate as batch_idx, F_plate:
            self.batch_idx = batch_idx.cpu()
            pyro.sample(
                "background", dist.Delta(
                    param(f"{prefix}/b_loc")[batch_idx]))
                #"background", dist.Gamma(
                #    param(f"{prefix}/b_loc")[batch_idx]
                #    * param(f"{prefix}/b_beta")[batch_idx],
                #    param(f"{prefix}/b_beta")[batch_idx]))
            with K_plate:
                pyro.sample(
                    "height", dist.Delta(
                        param(f"{prefix}/h_loc")[batch_idx]))
                    #"height", dist.Gamma(
                    #    param(f"{prefix}/h_loc")[batch_idx]
                    #    * param(f"{prefix}/h_beta")[batch_idx],
                    #    param(f"{prefix}/h_beta")[batch_idx]))
                pyro.sample(
                    "width", ScaledBeta(
                        param(f"{prefix}/w_mode")[batch_idx],
                        param(f"{prefix}/w_size")[batch_idx],
                        0.5, 2.5))
                pyro.sample(
                    "x0", ScaledBeta(
                        param(f"{prefix}/x_mode")[batch_idx],
                        param(f"{prefix}/size")[batch_idx],
                        -(data.D+3)/2, data.D+3))
                pyro.sample(
                    "y0", ScaledBeta(
                        param(f"{prefix}/y_mode")[batch_idx],
                        param(f"{prefix}/size")[batch_idx],
                        -(data.D+3)/2, data.D+3))

    def spot_parameters(self, data, prefix):
        param(f"{prefix}/background_loc",
              torch.ones(data.N, 1, 1, 1, 1) * 50.,
              constraint=constraints.positive)
        param(f"{prefix}/b_loc",
              torch.ones(data.N, data.F, 1, 1, 1) * 50.,
              constraint=constraints.positive)
        param(f"{prefix}/b_beta",
              torch.ones(data.N, data.F, 1, 1, 1) * 30,
              constraint=constraints.positive)
        param(f"{prefix}/h_loc",
              torch.ones(data.N, data.F, 1, 1, self.K) * 1000.,
              constraint=constraints.positive)
        param(f"{prefix}/h_beta",
              torch.ones(data.N, data.F, 1, 1, self.K),
              constraint=constraints.positive)
        param(f"{prefix}/w_mode",
              torch.ones(data.N, data.F, 1, 1, self.K) * 1.3,
              constraint=constraints.interval(0.5, 3.))
        param(f"{prefix}/w_size",
              torch.ones(data.N, data.F, 1, 1, self.K) * 100.,
              constraint=constraints.greater_than(2.))
        param(f"{prefix}/x_mode",
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(data.D+3)/2, (data.D+3)/2))
        param(f"{prefix}/y_mode",
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(data.D+3)/2, (data.D+3)/2))
        size = torch.ones(data.N, data.F, 1, 1, self.K) * 5.
        size[..., 0] = ((data.D+3) / (2*0.5)) ** 2 - 1
        param(f"{prefix}/size",
              size, constraint=constraints.greater_than(2.))

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+3)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        param("background_beta", torch.tensor([1.]),
              constraint=constraints.positive)
        param("height_loc", torch.tensor([1000.]),
              constraint=constraints.positive)
        param("height_beta", torch.tensor([0.01]),
              constraint=constraints.positive)
        param("width_mode", torch.tensor([1.3]),
              constraint=constraints.interval(0.5, 3.))
        param("width_size",
              torch.tensor([10.]), constraint=constraints.positive)
        param("pi", torch.ones(2), constraint=constraints.simplex)
        param("lamda", torch.tensor([0.1]), constraint=constraints.positive)

        if self.control:
            self.offset_max = torch.where(
                self.data[:].min() < self.control[:].min(),
                self.data[:].min() - 0.1,
                self.control[:].min() - 0.1)
        else:
            self.offset_max = self.data[:].min() - 0.1
        param("offset", self.offset_max-50,
              constraint=constraints.interval(0, self.offset_max))
        param("gain", torch.tensor(5.), constraint=constraints.positive)
