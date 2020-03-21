import torch
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate, infer_discrete
from pyro import param
from pyro import poutine
from pyro.contrib.autoname import scope
from pyro.ops.indexing import Vindex
from cosmos.models.helper import ScaledBeta
import torch.distributions.constraints as constraints

from cosmos.models.model import Model
from cosmos.models.helper import pi_m_calc, pi_theta_calc, trans_calc, init_calc


class HMM(Model):
    """ Hidden-Markov Model """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "hmm"
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

    @poutine.block(hide=["width_mode", "width_size"])
    def model(self):
        self.model_parameters(self.data)

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
                    * param("background_beta"), param("background_beta")).expand([len(batch_idx), data.F]).to_event(1))
            height = pyro.sample(
                "height", dist.Gamma(
                    param("height_loc") * param("height_beta"),
                    param("height_beta")).expand([data.F, self.K]).to_event(2))
            width = pyro.sample(
                "width", ScaledBeta(
                    param("width_mode"),
                    param("width_size"), 0.5, 2.5).expand([data.F, self.K]).to_event(2))

            #xy_dist = ScaledBeta(
            #        0, self.size, -(data.D+1)/2, data.D+1).expand([data.F, 8, self.K]).to_event(1)
            x = pyro.sample(
                "x", ScaledBeta(
                    0, self.size, -(data.D+1)/2, data.D+1).expand([data.F, self.K, 3]).to_event(3))
            y = pyro.sample(
                "y", ScaledBeta(
                    0, self.size, -(data.D+1)/2, data.D+1).expand([data.F, self.K, 3]).to_event(3))

            width = width * 2.5 + 0.5
            x = x * (data.D+1) - (data.D+1)/2
            y = y * (data.D+1) - (data.D+1)/2
            x = x[..., [0, 0, 1, 0, 2, 0, 1, 2]].permute(3, 0, 1, 2)
            y = y[..., [0, 0, 1, 0, 2, 0, 1, 2]].permute(3, 0, 1, 2)
            m_mask = self.m_matrix.bool().reshape(8, 1, 1, self.K)

            locs = data.loc(height, width, x, y, background, batch_idx, m_mask).permute(1, 2, 0, 3, 4)
            d_dist = self.CameraUnit(
                            locs, param("gain"), param("offset")).to_event(2)
            init = init_calc(param("pi"), param("lamda"), self.K)
            trans = trans_calc(param("A"), param("lamda"), self.K)
            pyro.sample(
                "data", dist.DiscreteHMM(init, trans, d_dist),
                obs=data[batch_idx])

    def spot_guide(self, data):
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-1)

        with N_plate as batch_idx:
            self.batch_idx = batch_idx
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
                    -(data.D+1)/2, data.D+1).to_event(3))
            pyro.sample(
                "y", ScaledBeta(
                    param("y_mode")[batch_idx],
                    param("size")[batch_idx],
                    -(data.D+1)/2, data.D+1).to_event(3))

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
              torch.zeros(data.N, data.F, self.K, 3),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        param("y_mode",
              torch.zeros(data.N, data.F, self.K, 3),
              constraint=constraints.interval(-(data.D+1)/2, (data.D+1)/2))
        size = torch.ones(data.N, data.F, self.K, 3) * 5.
        size[..., 0, 1] = ((data.D+1) / (2*0.5)) ** 2 - 1
        size[..., 1, 2] = ((data.D+1) / (2*0.5)) ** 2 - 1
        param("size",
              size, constraint=constraints.greater_than(2.))

    def model_parameters(self, data):
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
        param("pi", torch.ones(2), constraint=constraints.simplex)
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


    #@infer_discrete(first_available_dim=-2, temperature=0)
    @config_enumerate
    def viterbi_decoder(self, data):
        N_plate = pyro.plate("N_plate", data.N, dim=-1)
        init = init_calc(param("pi"), param("lamda"), self.K)
        trans = trans_calc(param("A"), param("lamda"), self.K)
        with N_plate as batch_idx:
            state = 0
            states = []
            for f in pyro.markov(range(data.F)):
                background = pyro.sample(
                    f"background_{f}", dist.Gamma(
                        param("b_loc")[batch_idx, f]
                        * param("b_beta")[batch_idx, f],
                        param("b_beta")[batch_idx, f]))

                height = pyro.sample(
                    f"height_{f}", dist.Gamma(
                        param("h_loc")[batch_idx, f]
                        * param("h_beta")[batch_idx, f],
                        param("h_beta")[batch_idx, f]).to_event(1))
                width = pyro.sample(
                    f"width_{f}", ScaledBeta(
                        param("w_mode")[batch_idx, f],
                        param("w_size")[batch_idx, f],
                        0.5, 2.5).to_event(1))
                x = pyro.sample(
                    f"x_{f}", ScaledBeta(
                        param("x_mode")[batch_idx, f],
                        param("size")[batch_idx, f],
                        -(data.D+1)/2, data.D+1).to_event(2))
                y = pyro.sample(
                    f"y_{f}", ScaledBeta(
                        param("y_mode")[batch_idx, f],
                        param("size")[batch_idx, f],
                        -(data.D+1)/2, data.D+1).to_event(2))

                state = pyro.sample(
                    f"state_{f}", dist.Categorical(Vindex(trans)[state, :]))

                width = width * 2.5 + 0.5
                x = x * (data.D+1) - (data.D+1)/2
                y = y * (data.D+1) - (data.D+1)/2
                x = x[..., [0, 0, 1, 0, 2, 0, 1, 2]]
                y = y[..., [0, 0, 1, 0, 2, 0, 1, 2]]
                y = Vindex(y)[..., :, state]
                x = Vindex(x)[..., :, state]
                m_mask = Vindex(self.m_matrix)[state, :].bool()

                locs = data.loc(height, width, x, y, background, batch_idx, m_mask, f)
                d_dist = self.CameraUnit(
                                locs, param("gain"), param("offset")).to_event(2)
                pyro.sample(
                    f"data_{f}", d_dist, obs=data[batch_idx, f])
                states.append(state)
        return states
