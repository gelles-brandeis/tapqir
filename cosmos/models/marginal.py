import torch
import pyro
import numpy as np
import os
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro import param
from pyro import poutine
from pyro.infer import infer_discrete
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.contrib.autoname import scope
from pyro.ops.indexing import Vindex
from cosmos.models.helper import ScaledBeta
import torch.distributions.constraints as constraints

from cosmos.models.model import Model
from cosmos.models.helper import pi_m_calc, pi_theta_calc
import cosmos
from git import Repo


class Marginal(Model):
    """ Track on-target Spot """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "marginal"
        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO)(
            max_plate_nesting=3, ignore_jit_warnings=True)
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

        #self.data.offset_weights, self.data.offset_vals = np.histogram(self.data.offset.cpu(), bins=20, density=True)
        #self.data.offset_weights = torch.log(torch.tensor(self.data.offset_weights))
        #self.data.offset_vals = torch.tensor(self.data.offset_vals[:-1])

        #self.control.offset_weights, self.control.offset_vals = np.histogram(self.control.offset.cpu(), bins=20, density=True)
        #self.control.offset_weights = torch.log(torch.tensor(self.control.offset_weights))
        #self.control.offset_vals = torch.tensor(self.control.offset_vals[:-1])
        self.offset_stat = self.data.offset.mean()
        self.data.data = torch.max(self.offset_stat+0.1, self.data.data)
        if control:
            self.control.data = torch.max(self.offset_stat+0.1, self.control.data)


    @poutine.block(hide=["width_mode", "width_size"])
    #@poutine.block(hide=["width_mode", "width_size", "height_loc", "height_beta"])
    @config_enumerate
    def model(self):
        self.model_parameters()
        pi_z = pyro.sample("pi_z", dist.Dirichlet(0.5 * torch.ones(self.S+1)))
        lamda = pyro.sample("lamda_j", dist.Dirichlet(0.5 * torch.ones(self.S+1)))
        data_pi_m = pi_m_calc(lamda, self.S)
        control_pi_m = pi_m_calc(lamda, self.S)
        pi_theta = pi_theta_calc(pi_z, self.K, self.S)
        #data_pi_m = pi_m_calc(pi_z, lamda, self.K)
        #control_pi_m = pi_m_calc(torch.tensor([1., 0.]), lamda, self.K)
        #pi_theta = pi_theta_calc(pi_z, lamda, self.K,)

        with scope(prefix="d"):
            self.spot_model(self.data, data_pi_m, pi_theta, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_model(self.control, control_pi_m, None, prefix="c")

    def guide(self):
        self.guide_parameters()
        pyro.sample("pi_z", dist.Dirichlet(param("pi") * param("size_z")))
        pyro.sample("lamda_j", dist.Dirichlet(param("lamda") * param("size_lamda")))

        with scope(prefix="d"):
            self.spot_guide(self.data, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_guide(self.control, prefix="c")

    def spot_model(self, data, pi_m, pi_theta, prefix):
        K_plate = pyro.plate("K_plate", self.K, dim=-3)
        N_plate = pyro.plate("N_plate", data.N, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, dim=-1)

        with N_plate as batch_idx, F_plate:
            background = pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/background_loc")[batch_idx]
                    * param("background_beta"), param("background_beta")))

            #if pi_theta is not None:
            #    theta = pyro.sample("theta", dist.Categorical(pi_theta))
            #else:
            #    theta = 0
            #theta_mask = Vindex(self.theta_matrix)[..., theta]
            #m_mask = Vindex(self.m_matrix)[..., theta]

            #m = pyro.sample("m", dist.Categorical(pi_m))
            #if pi_theta is not None:
            #    theta = pyro.sample("theta", dist.Categorical(pi_theta[m]))
            #    theta_mask = Vindex(self.theta_matrix)[..., theta]
            #else:
            #    theta_mask = 0

            if pi_theta is not None:
                theta = pyro.sample("theta", dist.Categorical(pi_theta))
            else:
                theta = 0
            theta_mask = Vindex(self.theta_matrix)[..., theta]
            m = pyro.sample("m", dist.Categorical(Vindex(pi_m)[theta]))
            m_mask = Vindex(self.m_matrix)[..., m]

            #height_loc = torch.cat((torch.tensor([0.01]), param("height_loc")), 0)
            #height_beta = torch.cat((torch.tensor([100.]), param("height_beta")), 0)
            #height_loc = torch.cumsum(param("height_loc"), 0)
            height_loc = param("height_loc")
            height_beta = param("height_beta")
            with K_plate:
                #m_mask = pyro.sample("m", dist.Categorical(Vindex(pi_m)[m_mask]))
                #with poutine.mask(mask=m_mask>0):
                height = pyro.sample(
                    "height", dist.Gamma(
                        height_loc * height_beta,
                        height_beta))
                        #height_loc[m_mask] * height_beta[m_mask],
                        #height_beta[m_mask]))
                width = pyro.sample(
                    "width", ScaledBeta(
                        param("width_mode"),
                        param("width_size"), 0.5, 2.5))
                x = pyro.sample(
                    "x", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1))
                y = pyro.sample(
                    "y", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1))

            height = height.masked_fill(m_mask==0, 0.)
            width = width * 2.5 + 0.5
            x = x * (data.D+1) - (data.D+1)/2
            y = y * (data.D+1) - (data.D+1)/2

            locs = data.loc(height, width, x, y, background, batch_idx)
            #offset = dist.Empirical(data.offset_vals, data.offset_weights).sample()
            #offset = 100.
            offset = self.offset_stat
            pyro.sample(
                "data", self.CameraUnit(
                    locs, param("gain"), offset).to_event(2),
                    #locs, param("gain"), param("offset")).to_event(2),
                obs=data[batch_idx])

    def spot_guide(self, data, prefix):
        K_plate = pyro.plate("K_plate", self.K, dim=-3)
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, dim=-1)

        with N_plate as batch_idx, F_plate:
            if prefix == "d":
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
                pyro.sample(
                    "x", ScaledBeta(
                        param(f"{prefix}/x_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1))
                pyro.sample(
                    "y", ScaledBeta(
                        param(f"{prefix}/y_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1))

    def guide_parameters(self):
        param("pi", torch.ones(self.S+1), constraint=constraints.simplex)
        #param("pi", torch.tensor([0.1, 0.9]), constraint=constraints.simplex)
        param("lamda", torch.ones(self.S+1), constraint=constraints.simplex)
        #param("lamda", torch.tensor([0.1]), constraint=constraints.positive)
        param("size_z", torch.tensor([1000.]), constraint=constraints.positive)
        param("size_lamda", torch.tensor([1000.]), constraint=constraints.positive)
        self.spot_parameters(self.data, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, prefix="c")

    def spot_parameters(self, data, prefix):
        param(f"{prefix}/background_loc",
              data[:].mean(dim=(1,2,3)).reshape(data.N, 1) - data.offset.mean(),
              #torch.ones(data.N, 1) * 50.,
              constraint=constraints.positive)
        param(f"{prefix}/b_loc",
              torch.ones(data.N, data.F) * 50.,
              constraint=constraints.positive)
        param(f"{prefix}/b_beta",
              torch.ones(data.N, data.F) * 30,
              constraint=constraints.positive)
        param(f"{prefix}/h_loc",
              #torch.ones(self.K, data.N, data.F) * 300.,
              torch.ones(self.K, data.N, data.F) * 1000.,
              constraint=constraints.positive)
        param(f"{prefix}/h_beta",
              torch.ones(self.K, data.N, data.F),
              constraint=constraints.positive)
        param(f"{prefix}/w_mode",
              torch.ones(self.K, data.N, data.F) * 1.3,
              constraint=constraints.interval(0.5, 3.))
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

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+1)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        #param("height_loc", torch.tensor([100., 900., 1000.])[:self.S+1],
        #param("height_loc", torch.tensor([300.]),
        param("height_loc", torch.tensor([1000.]),
              constraint=constraints.positive)
        param("height_beta", torch.tensor([0.01]),
              constraint=constraints.positive)
        #param("height_loc", torch.tensor([1000.]),
        #      constraint=constraints.positive)
        #param("height_beta", torch.tensor([0.01]),
        #      constraint=constraints.positive)
        param("background_beta", torch.tensor([30.]),
              constraint=constraints.positive)
        param("width_mode", torch.tensor([1.3]),
              constraint=constraints.interval(0.5, 3.))
        param("width_size",
              torch.tensor([10.]), constraint=constraints.positive)

        if self.control:
            self.offset_max = torch.where(
                self.data[:].min() < self.control[:].min(),
                self.data[:].min() - 0.1,
                self.control[:].min() - 0.1)
        else:
            self.offset_max = self.data[:].min() - 0.1
        param("offset", self.offset_max-50,
              constraint=constraints.interval(0, self.offset_max))
        #param("offset", torch.tensor([90.]), constraint=constraints.positive)
        param("gain", torch.tensor(7.), constraint=constraints.positive)

    def infer(self):
        guide_trace = poutine.trace(self.guide).get_trace()
        trained_model = poutine.replay(
            poutine.enum(self.model, first_available_dim=-4), trace=guide_trace)
        inferred_model = infer_discrete(
            trained_model, temperature=0, first_available_dim=-4)
        trace = poutine.trace(inferred_model).get_trace()
        self.predictions["z"][self.batch_idx] = (
            trace.nodes["d/theta"]["value"] > 0) \
            .cpu().data.squeeze()
        self.predictions["m"][self.batch_idx] = \
            Vindex(self.m_matrix)[..., trace.nodes["d/m"]["value"]].cpu().data.permute(1, 2, 0)
        np.save(os.path.join(self.path, "predictions.npy"),
                self.predictions)
