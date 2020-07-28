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

        repo = Repo(cosmos.__path__[0], search_parent_directories=True)
        version = repo.git.describe()
        param_path = os.path.join(
            path, "runs",
            "{}{}".format("feature", version),
            "{}".format("jit" if self.jit else "nojit"),
            "lr{}".format(self.lr), "{}".format(self.optim_fn.__name__),
            "{}".format(self.n_batch))
        pyro.get_param_store().load(
            os.path.join(param_path, "params"),
            map_location=self.data.device)

    @poutine.block(hide=["width_mode", "width_size"])
    @config_enumerate
    def model(self):
        self.model_parameters()
        pi_z = pyro.sample("pi_z", dist.Dirichlet(0.5 * torch.ones(self.S+1)))
        lamda = pyro.sample("lamda_j", dist.Dirichlet(0.5 * torch.ones(self.S+1)))
        data_pi_m = pi_m_calc(lamda, self.S)
        control_pi_m = pi_m_calc(lamda, self.S)
        pi_theta = pi_theta_calc(pi_z, self.K, self.S)

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

            if pi_theta is not None:
                theta = pyro.sample("theta", dist.Categorical(pi_theta))
            else:
                theta = 0
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
                x = pyro.sample(
                    "x", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1))
                y = pyro.sample(
                    "y", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1))

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
        param("lamda", torch.ones(self.S+1), constraint=constraints.simplex)
        param("size_z", torch.tensor([1000.]), constraint=constraints.positive)
        param("size_lamda", torch.tensor([1000.]), constraint=constraints.positive)
        self.spot_parameters(self.data, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, prefix="c")

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
        # self.predictions["m"][self.batch_idx] = \
        #     trace.nodes["d/m"]["value"].cpu().data.squeeze()
        np.save(os.path.join(self.path, "predictions.npy"),
                self.predictions)