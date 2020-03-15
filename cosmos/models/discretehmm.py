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
from cosmos.models.helper import pi_m_calc, pi_theta_calc, trans_m_calc, trans_theta_calc


class HMM(Model):
    """ Hidden-Markov Model """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "hmm"
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

    @poutine.block(hide=["width_mode", "width_size"])
    def model(self):
        self.model_parameters()

        self.spot_model(self.data)

        if self.control:
            self.spot_model(self.control)

    def guide(self):
        self.guide_parameters()
        self.spot_guide(self.data)

        if self.control:
            self.spot_guide(self.control)

    def guide_parameters(self):
        self.spot_parameters(self.data)
        if self.control:
            self.spot_parameters(self.control)


    def spot_model(self, data):
        N_plate = pyro.plate("N_plate", data.N, dim=-1)
        with N_plate as batch_idx:
            background = pyro.sample(
                "background", dist.Gamma(
                    param("background_loc")[batch_idx]
                    * param("background_beta"), param("background_beta")).expand([-1, data.F]).to_event(1))
            height = pyro.sample(
                "height", dist.Gamma(
                    param("height_loc") * param("height_beta"),
                    param("height_beta")).expand([data.F, self.K]).to_event(2))
            width = pyro.sample(
                "width", ScaledBeta(
                    param("width_mode"),
                    param("width_size"), 0.5, 2.5).expand([data.F, self.K]).to_event(2))

            xy_dist = ScaledBeta(
                    0, self.size, -(data.D+1)/2, data.D+1).expand([1, -1, -1]).to_event(1)
            init_theta = pi_theta_calc(param("A"), self.K)
            trans_theta = trans_theta_calc(param("A"), self.K)
            x = pyro.sample(
                "x", dist.DiscreteHMM(init_theta, trans_theta, xy_dist))
            y = pyro.sample(
                "y", dist.DiscreteHMM(init_theta, trans_theta, xy_dist))

            width = width * 2.5 + 0.5
            x = x * (data.D+1) - (data.D+1)/2
            y = y * (data.D+1) - (data.D+1)/2

            locs = data.loc(height, width, x, y, background, batch_idx)
            d_dist = self.CameraUnit(
                            locs, param("gain"), param("offset")).to_event(2)
            init_m = pi_m_calc(param("A"), param("lamda"), self.K)
            trans_m = trans_m_calc(param("A"), param("lamda"), self.K)
            pyro.sample(
                "data", dist.DiscreteHMM(init_m, trans_m, d_dist),
                obs=data[batch_idx])

    def spot_guide(self, data):
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-1)

        with N_plate as batch_idx:
            self.batch_idx = batch_idx.cpu()
            pyro.sample(
                "background", dist.Gamma(
                    param("b_loc")[batch_idx]
                    * param("b_beta")[batch_idx],
                    param("b_beta")[batch_idx]).to_event(1))

            pyro.sample(
                "height", dist.Gamma(
                    param("h_loc")[batch_idx]
                    * param("h_beta")[batch_idx],
                    param("h_beta")[batch_idx]).to_event(2))
            pyro.sample(
                "width", ScaledBeta(
                    param("w_mode")[batch_idx],
                    param("w_size")[batch_idx],
                    0.5, 2.5).to_event(2))
            pyro.sample(
                "x", ScaledBeta(
                    param("x_mode")[batch_idx],
                    param("size")[batch_idx],
                    -(data.D+1)/2, data.D+1).to_event(2))
            pyro.sample(
                "y", ScaledBeta(
                    param("y_mode")[batch_idx],
                    param("size")[batch_idx],
                    -(data.D+1)/2, data.D+1).to_event(2))

    def spot_parameters(self, data):
        param("b_loc",
              torch.ones(data.N, data.F) * 50.,
              constraint=constraints.positive)
        param("b_beta",
              torch.ones(data.N, data.F) * 30,
              constraint=constraints.positive)
        param("h_loc",
              torch.ones(data.N, data.F, self.K) * 1000.,
              constraint=constraints.positive)
        param("h_beta",
              torch.ones(data.N, data.F, self.K),
              constraint=constraints.positive)
        param("w_mode",
              torch.ones(data.N, data.F, self.K) * 1.3,
              constraint=constraints.interval(0.5, 3.))
        param("w_size",
              torch.ones(data.N, data.F, self.K) * 100.,
              constraint=constraints.greater_than(2.))
        param("x_mode",
              torch.zeros(data.N, data.F, self.K),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        param("y_mode",
              torch.zeros(data.N, data.F, self.K),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        size = torch.ones(data.N, data.F, self.K) * 5.
        size[..., 0] = ((data.D+1) / (2*0.5)) ** 2 - 1
        param("size",
              size, constraint=constraints.greater_than(2.))

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+1)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        param("background_loc",
              torch.ones(data.N, 1) * 50.,
              constraint=constraints.positive)
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
        param("A", torch.ones(2, 2), constraint=constraints.simplex)
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
