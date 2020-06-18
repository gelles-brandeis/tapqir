import torch
import pyro
import numpy as np
import os
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro import param
from pyro import poutine
from pyro.infer import SVI, infer_discrete
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.contrib.autoname import scope
from pyro.ops.indexing import Vindex
from cosmos.models.helper import ScaledBeta
import torch.distributions.constraints as constraints

from cosmos.models.model import Model
from cosmos.models.helper import pi_m_calc, pi_theta_calc
from cosmos.models.helper import z_probs_calc, k_probs_calc
import cosmos
from git import Repo


class Tracker(Model):
    """ Track on-target Spot """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "tracker"
        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO)(
            max_plate_nesting=3, ignore_jit_warnings=True)
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

        repo = Repo(cosmos.__path__[0], search_parent_directories=True)
        version = repo.git.describe()
        param_path = os.path.join(
            path, "runs",
            "{}{}".format("marginal", version),
            "{}".format("jit" if self.jit else "nojit"),
            "lr{}".format(self.lr), "{}".format(self.optim_fn.__name__),
            "{}".format(self.n_batch))
        pyro.get_param_store().load(
            os.path.join(param_path, "params"),
            map_location=self.data.device)
        #self.offset_stat = self.data.offset.mean()
        #self.data.data = torch.max(self.offset_stat+0.1, self.data.data)
        #if control:
        #    self.control.data = torch.max(self.offset_stat+0.1, self.control.data)

    #@poutine.block(hide=["width_mode", "width_size"])
    @poutine.block(hide_types=["param"])
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

    @poutine.block(expose_types=["sample"], expose=["d/theta_probs", "d/m_probs"])
    @config_enumerate
    def guide(self):
        self.guide_parameters()
        pyro.sample("pi_z", dist.Dirichlet(param("pi") * param("size_z")))
        pyro.sample("lamda_j", dist.Dirichlet(param("lamda") * param("size_lamda")))

        with scope(prefix="d"):
            self.spot_guide(self.data, True, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_guide(self.control, False, prefix="c")

    def guide_parameters(self):
        self.spot_parameters(self.data, True, True, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, True, False, prefix="c")


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
            m = pyro.sample("m", dist.Categorical(Vindex(pi_m)[theta]))
            m_mask = Vindex(self.m_matrix)[..., m]

            height_loc = param("height_loc")
            height_beta = param("height_beta")
            with K_plate:
                #height = pyro.sample(
                #    "height", dist.Gamma(
                #        param("height_loc")[m_mask] * param("height_beta")[m_mask],
                #        param("height_beta")[m_mask]))
                height = pyro.sample(
                    "height", dist.Gamma(
                        height_loc * height_beta,
                        height_beta))
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
            #offset = self.offset_stat
            pyro.sample(
                "data", self.CameraUnit(
                    #locs, param("gain"), offset).to_event(2),
                    locs, param("gain"), param("offset")).to_event(2),
                obs=data[batch_idx])

    def spot_guide(self, data, theta, prefix):
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

            if theta:
                theta = pyro.sample("theta", dist.Categorical(
                    param(f"{prefix}/theta_probs")[batch_idx]))
            else:
                theta = 0
            pyro.sample("m", dist.Categorical(
                Vindex(param(f"{prefix}/m_probs")[batch_idx])[..., theta, :]))

            with K_plate:
                #pyro.sample("m", dist.Categorical(
                #    Vindex(param(f"{prefix}/m_probs")[:, batch_idx])[..., m_mask, :]))

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

    def spot_parameters(self, data, m, theta, prefix):
        if m:
            #m_probs = torch.zeros(self.K, data.N, data.F, self.S+1, self.S+1)
            m_probs = torch.ones(data.N, data.F, self.K+1, 2**self.K)
            #m_probs = pi_m_calc(param("lamda"), self.S).repeat(data.N, data.F, 1, 1)
            m_probs[..., 1, 0] = 0
            m_probs[..., 1, 2] = 0
            m_probs[..., 2, 0] = 0
            m_probs[..., 2, 1] = 0
            param(f"{prefix}/m_probs",
                  m_probs,
                  constraint=constraints.simplex)
        if theta:
            #theta_probs = pi_theta_calc(param("pi"), self.K, self.S).repeat(data.N, data.F, 1)
            theta_probs = torch.ones(
                data.N, data.F, self.S*self.K+1)
            param(f"{prefix}/theta_probs", theta_probs,
                  constraint=constraints.simplex)

    def model_parameters(self):
        pass

    def infer(self):
        z_probs = z_probs_calc(
            pyro.param("d/m_probs"), pyro.param("d/theta_probs"))
        k_probs = k_probs_calc(pyro.param("d/m_probs"), pyro.param("d/theta_probs"))
        self.predictions["z_prob"] = z_probs.squeeze()
        self.predictions["z"] = \
            self.predictions["z_prob"] > 0.5
        self.predictions["m_prob"] = k_probs.squeeze()
        self.predictions["m"] = self.predictions["m_prob"] > 0.5
        np.save(os.path.join(self.path, "predictions.npy"),
                self.predictions)
