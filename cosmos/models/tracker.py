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


class Tracker(Model):
    """ Track on-target Spot """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "tracker"
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

    #@poutine.block(hide=["width_mode", "width_size"])
    @poutine.block(hide_types=["param"])
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

    @poutine.block(expose_types=["sample"], expose=["d/theta_probs", "d/m_probs"])
    @config_enumerate
    def guide(self):
        self.guide_parameters()
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
        N_plate = pyro.plate("N_plate", data.N, dim=-2)
        F_plate = pyro.plate("F_plate", data.F, dim=-1)

        with N_plate as batch_idx, F_plate:
            background = pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/background_loc")[batch_idx]
                    * param("background_beta"), param("background_beta")))

            m = pyro.sample("m", dist.Categorical(pi_m))

            if pi_theta is not None:
                theta = pyro.sample("theta", dist.Categorical(pi_theta[m]))
                theta_mask = Vindex(self.theta_matrix)[theta]
            else:
                theta_mask = 0
            m_mask = Vindex(self.m_matrix)[m, :].bool()

            height = pyro.sample(
                "height", dist.Gamma(
                    param("height_loc") * param("height_beta"),
                    param("height_beta")).expand([self.K]).to_event(1))
            width = pyro.sample(
                "width", ScaledBeta(
                    param("width_mode"),
                    param("width_size"), 0.5, 2.5).expand([self.K]).to_event(1))
            x = pyro.sample(
                "x", ScaledBeta(
                    0, self.size[theta_mask], -(data.D+1)/2, data.D+1).to_event(1))
            y = pyro.sample(
                "y", ScaledBeta(
                    0, self.size[theta_mask], -(data.D+1)/2, data.D+1).to_event(1))

            width = width * 2.5 + 0.5
            x = x * (data.D+1) - (data.D+1)/2
            y = y * (data.D+1) - (data.D+1)/2

            locs = data.loc(height, width, x, y, background, batch_idx, m_mask)
            pyro.sample(
                "data", self.CameraUnit(
                    locs, param("gain"), param("offset")).to_event(2),
                obs=data[batch_idx])

    def spot_guide(self, data, theta, prefix):
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

            m = pyro.sample("m", dist.Categorical(
                param(f"{prefix}/m_probs")[batch_idx]))
            if theta:
                pyro.sample("theta", dist.Categorical(
                    Vindex(param(
                        f"{prefix}/theta_probs")[batch_idx])[..., m, :]))

            pyro.sample(
                "height", dist.Gamma(
                    param(f"{prefix}/h_loc")[batch_idx]
                    * param(f"{prefix}/h_beta")[batch_idx],
                    param(f"{prefix}/h_beta")[batch_idx]).to_event(1))
            pyro.sample(
                "width", ScaledBeta(
                    param(f"{prefix}/w_mode")[batch_idx],
                    param(f"{prefix}/w_size")[batch_idx],
                    0.5, 2.5).to_event(1))
            pyro.sample(
                "x", ScaledBeta(
                    param(f"{prefix}/x_mode")[batch_idx],
                    param(f"{prefix}/size")[batch_idx],
                    -(data.D+1)/2, data.D+1).to_event(1))
            pyro.sample(
                "y", ScaledBeta(
                    param(f"{prefix}/y_mode")[batch_idx],
                    param(f"{prefix}/size")[batch_idx],
                    -(data.D+1)/2, data.D+1).to_event(1))

    def spot_parameters(self, data, m, theta, prefix):
        if m:
            param(f"{prefix}/m_probs",
                  torch.ones(data.N, data.F, 2**self.K),
                  #torch.ones(data.N, data.F, 1, 1, 1, 2**self.K),
                  constraint=constraints.simplex)
        if theta:
            theta_probs = torch.ones(
                data.N, data.F, 2**self.K, 2)
            #    data.N, data.F, 1, 1, 1, 2**self.K, self.K+1)
            theta_probs[..., 0, 1] = 0
            #theta_probs[..., 1, 2] = 0
            theta_probs[..., 2, 1] = 0
            param(f"{prefix}/theta_probs", theta_probs,
                  constraint=constraints.simplex)

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+1)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        pass
