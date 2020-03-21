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
from cosmos.models.helper import pi_m_calc, pi_theta_calc, A_m_calc, A_theta_calc


class HMM(Model):
    """ Hidden-Markov Model """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "hmm"
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

    @poutine.block(hide=["width_mode", "width_size"])
    @config_enumerate
    def model(self):
        self.model_parameters()
        data_A_m = A_m_calc(param("A"), param("lamda"), self.K)
        control_pi_m = pi_m_calc(torch.tensor([1., 0.]), param("lamda"), self.K)
        control_A_m = control_pi_m.unsqueeze(dim=0)
        A_theta = A_theta_calc(param("A"), param("lamda"), self.K)

        #with scope(prefix="d"):
        self.spot_model(self.data, data_A_m, A_theta, prefix="d")

        if self.control:
            #with scope(prefix="c"):
            self.spot_model(self.control, control_A_m, None, prefix="c")

    def guide(self):
        self.guide_parameters()
        #with scope(prefix="d"):
        self.spot_guide(self.data, False, prefix="d")

        if self.control:
            #with scope(prefix="c"):
            self.spot_guide(self.control, False, prefix="c")

    def guide_parameters(self):
        self.spot_parameters(self.data, True, True, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, True, False, prefix="c")


    def spot_model(self, data, A_m, A_theta, prefix):
        N_plate = pyro.plate("N_plate", data.N, dim=-4)
        I_plate = pyro.plate("I_plate", data.D, dim=-3)
        J_plate = pyro.plate("J_plate", data.D, dim=-2)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)


        with N_plate as batch_idx:
            theta = 0
            for f in pyro.markov(range(data.F)):
                background = pyro.sample(
                    f"background_{f}", dist.Gamma(
                        param(f"{prefix}/background_loc")[batch_idx]
                        * param("background_beta"), param("background_beta")))

                m = pyro.sample(f"m_{f}", dist.Categorical(
                    Vindex(A_m)[theta, :]))
                if A_theta is not None:
                    theta = pyro.sample(f"theta_{f}", dist.Categorical(
                        Vindex(A_theta)[theta, m]))
                    theta_mask = self.theta_matrix[theta.squeeze(dim=-1)]
                else:
                    theta_mask = 0
                m_mask = self.m_matrix[m.squeeze(dim=-1)].bool()

                with K_plate:
                    height = pyro.sample(
                        f"height_{f}", dist.Gamma(
                            param("height_loc") * param("height_beta"),
                            param("height_beta")))
                    width = pyro.sample(
                        f"width_{f}", ScaledBeta(
                            param("width_mode"),
                            param("width_size"), 0.5, 2.5))
                    x = pyro.sample(
                        f"x_{f}", ScaledBeta(
                            0, self.size[theta_mask], -(data.D+1)/2, data.D+1))
                    y = pyro.sample(
                        f"y_{f}", ScaledBeta(
                            0, self.size[theta_mask], -(data.D+1)/2, data.D+1))

                width = width * 2.5 + 0.5
                x = x * (data.D+1) - (data.D+1)/2
                y = y * (data.D+1) - (data.D+1)/2

                locs = data.loc(m_mask, height, width,
                                x, y, background, batch_idx, f)
                with I_plate, J_plate:
                    pyro.sample(
                        f"data_{f}", self.CameraUnit(
                            locs, param("gain"), param("offset")),
                        obs=data[batch_idx, f].unsqueeze(dim=-1))

    def spot_guide(self, data, theta, prefix):
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-4)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)

        with N_plate as batch_idx:
            self.batch_idx = batch_idx.cpu()
            #m = torch.zeros(len(batch_idx), data.F, 1, 1, 1).long()
            #theta = torch.zeros(len(batch_idx), data.F, 1, 1, 1).long()
            #t = theta[:, 0:1]
            for f in pyro.markov(range(data.F)):
                pyro.sample(
                    f"background_{f}", dist.Gamma(
                        param(f"{prefix}/b_loc")[batch_idx, f]
                        * param(f"{prefix}/b_beta")[batch_idx, f],
                        param(f"{prefix}/b_beta")[batch_idx, f]))
                #m[:, f:f+1] = pyro.sample(f"m_{f}", dist.Categorical(
                #    Vindex(param(f"{prefix}/m_probs")[batch_idx, f:f+1])
                #    [..., t, :]))
                #if theta:
                #    pyro.sample(f"theta_{f}", dist.Categorical(
                #        Vindex(param(
                #            f"{prefix}/theta_probs")[batch_idx, f:f+1])[..., t, m[:, f:f+1], :]))

                with K_plate:
                    pyro.sample(
                        f"height_{f}", dist.Gamma(
                            param(f"{prefix}/h_loc")[batch_idx, f]
                            * param(f"{prefix}/h_beta")[batch_idx, f],
                            param(f"{prefix}/h_beta")[batch_idx, f]))
                    pyro.sample(
                        f"width_{f}", ScaledBeta(
                            param(f"{prefix}/w_mode")[batch_idx, f],
                            param(f"{prefix}/w_size")[batch_idx, f],
                            0.5, 2.5))
                    pyro.sample(
                        f"x_{f}", ScaledBeta(
                            param(f"{prefix}/x_mode")[batch_idx, f],
                            param(f"{prefix}/size")[batch_idx, f],
                            -(data.D+1)/2, data.D+1))
                    pyro.sample(
                        f"y_{f}", ScaledBeta(
                            param(f"{prefix}/y_mode")[batch_idx, f],
                            param(f"{prefix}/size")[batch_idx, f],
                            -(data.D+1)/2, data.D+1))

    def spot_parameters(self, data, m, theta, prefix):
        param(f"{prefix}/background_loc",
              torch.ones(data.N, 1, 1, 1) * 50.,
              constraint=constraints.positive)
        if m:
            m_probs = torch.ones(data.N, data.F, 1, 1, 1, self.K+1, 2**self.K)
            param(f"{prefix}/m_probs", m_probs,
                  constraint=constraints.simplex)
        if theta:
            theta_probs = torch.ones(
                data.N, data.F, 1, 1, 1, self.K+1, 2**self.K, self.K+1)
            theta_probs[..., 0, 1:] = 0
            theta_probs[..., 1, 2] = 0
            theta_probs[..., 2, 1] = 0
            param(f"{prefix}/theta_probs", theta_probs,
                  constraint=constraints.simplex)
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
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        param(f"{prefix}/y_mode",
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        size = torch.ones(data.N, data.F, 1, 1, self.K) * 5.
        size[..., 0] = ((data.D+1) / (2*0.5)) ** 2 - 1
        param(f"{prefix}/size",
              size, constraint=constraints.greater_than(2.))

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+1)/(2*0.5))**2 - 1)]),
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
