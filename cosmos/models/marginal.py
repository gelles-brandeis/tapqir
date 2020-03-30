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
from pyro.distributions import TorchDistribution

from cosmos.models.model import Model
from cosmos.models.helper import pi_m_calc, pi_theta_calc, theta_trans_calc
import torch.nn as nn


class CatDistribution(TorchDistribution):
    """
    Concatenate multiple heterogeneous distributions.

    This is useful when multiple heterogeneous distributions
    depend on the same hidden state in DiscreteHMM.

    Example::

    :param
    """
    arg_constraints = {}  # nothing to be constrained

    def __init__(self, *dists):
        self.dists = dists
        batch_shape = self.dists[0].batch_shape
        event_shape = self.dists[0].event_shape + (len(self.dists),)
        super().__init__(batch_shape, event_shape)

    @property
    def has_rsample(self):
        return all(dist.has_rsample for dist in self.dists)

    def expand(self, batch_shape):
        dists = (dist.expand(batch_shape) for dist in self.dists)
        return type(self)(*dists)

    def sample(self, sample_shape=torch.Size()):
        result = tuple(dist.sample(sample_shape) for dist in self.dists)
        return torch.stack(result, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        result = tuple(dist.rsample(sample_shape) for dist in self.dists)
        return torch.stack(result, dim=-1)

    def log_prob(self, value):
        values = torch.unbind(value, dim=-1)
        log_probs = tuple(dist.log_prob(value) for dist, value in zip(self.dists, values))
        result = torch.sum(torch.stack(log_probs, -1), -1)
        return result

class EnumDistribution(TorchDistribution):
    arg_constraints = {}  # nothing to be constrained

    def __init__(self, dist, emission_logits):
        self.dist = dist
        self.emission_logits = emission_logits
        batch_shape = self.dist.batch_shape
        event_shape = self.dist.event_shape
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        value = value.unsqueeze(-1 - self.observation_dist.event_dim)
        obs_logits = self.dist.log_prob(value)
        result = obs_logits.unsqueeze(dim=-2) + self.emission_logits
        result = torch.logsumexp(result, -1)
        return result

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
        data_pi_m = pi_m_calc(param("lamda"), self.S)
        control_pi_m = pi_m_calc(param("lamda"), self.S)
        pi_theta = pi_theta_calc(param("pi"), self.K, self.S)

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
        K_plate = pyro.plate("K_plate", self.K, dim=-3)
        N_plate = pyro.plate("N_plate", data.N, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, dim=-1)

        with N_plate as batch_idx, F_plate:
            background = pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/background_loc")[batch_idx]
                    * param("background_beta"), param("background_beta")))


            theta = pyro.sample("theta", dist.Categorical(pi_theta))
            theta_mask = Vindex(self.theta_matrix)[..., theta]
            m_mask = Vindex(self.m_matrix)[..., theta]

            with K_plate:
                m_mask = pyro.sample("m", dist.Categorical(Vindex(pi_m)[m_mask]))
                height = pyro.sample(
                    "height", dist.Gamma(
                        param("height_loc")[m_mask] * param("height_beta")[m_mask],
                        param("height_beta")[m_mask]))
                width = pyro.sample(
                    "width", ScaledBeta(
                        param("width_mode"),
                        param("width_size"), 0.5, 2.5))
                #x = pyro.sample(
                #    "x", ScaledBeta(
                #        0, self.size[theta_mask], -(data.D+1)/2, data.D+1))
                #y = pyro.sample(
                #    "y", ScaledBeta(
                #        0, self.size[theta_mask], -(data.D+1)/2, data.D+1))
                #xy = pyro.sample(
                #    "xy", ScaledBeta(
                #        0, torch.stack([self.size[theta_mask], self.size[theta_mask]], dim=-1), -(data.D+1)/2, data.D+1).to_event(1))
                x_dist = ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1)
                y_dist = ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1)
                xy_dist = CatDistribution(x_dist, y_dist)
                xy = pyro.sample("xy", xy_dist)
                x, y = torch.unbind(xy, dim=-1)

            width = width * 2.5 + 0.5
            x = x * (data.D+1) - (data.D+1)/2
            y = y * (data.D+1) - (data.D+1)/2

            locs = data.loc(height, width, x, y, background, batch_idx)
            pyro.sample(
                "data", self.CameraUnit(
                    locs, param("gain"), param("offset")).to_event(2),
                obs=data[batch_idx])

    def spot_guide(self, data, prefix):
        K_plate = pyro.plate("K_plate", self.K, dim=-3)
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, dim=-1)

        with N_plate as batch_idx, F_plate:
            self.batch_idx = batch_idx.cpu()
            pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/b_loc")[batch_idx]
                    * param(f"{prefix}/b_beta")[batch_idx],
                    param(f"{prefix}/b_beta")[batch_idx]))
            with K_plate:
                pyro.sample(
                    "height", dist.Gamma(
                        param(f"{prefix}/h_loc")[:, batch_idx]
                        * param(f"{prefix}/h_beta")[:, batch_idx],
                        param(f"{prefix}/h_beta")[:, batch_idx]))
                pyro.sample(
                    "width", ScaledBeta(
                        param(f"{prefix}/w_mode")[:, batch_idx],
                        param(f"{prefix}/w_size")[:, batch_idx],
                        0.5, 2.5))
                #pyro.sample(
                #    "x", ScaledBeta(
                #        param(f"{prefix}/x_mode")[:, batch_idx],
                #        param(f"{prefix}/size")[:, batch_idx],
                #        -(data.D+1)/2, data.D+1))
                #pyro.sample(
                #    "y", ScaledBeta(
                #        param(f"{prefix}/y_mode")[:, batch_idx],
                #        param(f"{prefix}/size")[:, batch_idx],
                #        -(data.D+1)/2, data.D+1))
                #pyro.sample(
                #    "xy", ScaledBeta(
                #        torch.stack([param(f"{prefix}/x_mode")[:, batch_idx], param(f"{prefix}/y_mode")[:, batch_idx]], dim=-1),
                #        torch.stack([param(f"{prefix}/size")[:, batch_idx], param(f"{prefix}/size")[:, batch_idx]], dim=-1),
                #        -(data.D+1)/2, data.D+1).to_event(1))
                x_dist = ScaledBeta(
                        param(f"{prefix}/x_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1)
                y_dist = ScaledBeta(
                        param(f"{prefix}/y_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1)
                xy_dist = CatDistribution(x_dist, y_dist)
                pyro.sample("xy", xy_dist)

    def spot_parameters(self, data, prefix):
        pass

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+1)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        param("height_loc", torch.tensor([100., 1000., 2000.])[:self.S+1],
              constraint=constraints.positive)
        param("height_beta", torch.tensor([0.01, 0.01, 0.01])[:self.S+1],
              constraint=constraints.positive)
        param("pi", torch.ones(self.S+1), constraint=constraints.simplex)
        param("lamda", torch.ones(self.S+1), constraint=constraints.simplex)
