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
from cosmos.models.helper import pi_m_calc, pi_theta_calc, theta_trans_calc


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
        event_shape = self.dists[0].event_shape
        super().__init__(batch_shape, event_shape)

    @property
    def has_rsample(self):
        return all(dist.has_rsample for dist in self.dists)

    def expand(self, batch_shape):
        dists = (dist.expand(batch_shape) for dist in self.dists)
        return type(self)(*dists)

    def sample(self, sample_shape=torch.Size()):
        result = [dist.sample(sample_shape) for dist in self.dists]
        return result

    def rsample(self, sample_shape=torch.Size()):
        result = [dist.sample(sample_shape) for dist in self.dists]
        return result

    def log_prob(self, values):
        log_probs = [dist.log_prob(value) for dist, value in zip(self.dists, values)]
        result = torch.sum(torch.stack(log_probs, 0), 0)
        return result

class HMM(Model):
    """ Hidden-Markov Model """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "hmm"
        super().__init__(data, control, path,
                         K, lr, n_batch, jit, noise="GammaOffset")

    @poutine.block(hide=["width_mode", "width_size"])
    #@config_enumerate
    def model(self):
        self.model_parameters(self.data)

        self.spot_model(self.data, prefix="d")

        if self.control:
            self.spot_model(self.control, prefix="c")

    def guide(self):
        self.guide_parameters()
        self.spot_guide(self.data, prefix="d")

        if self.control:
            self.spot_guide(self.control, prefix="c")

    def guide_parameters(self):
        self.spot_parameters(self.data, prefix="d")
        if self.control:
            self.spot_parameters(self.control, prefix="c")


    def spot_model(self, data, prefix):
        K_plate = pyro.plate("K_plate", self.K, dim=-2)
        N_plate = pyro.plate("N_plate", data.N, dim=-1)

        with N_plate as batch_idx:
            background = pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/background_loc")[batch_idx]
                    * param("background_beta"), param("background_beta")).expand([len(batch_idx), data.F]).to_event(1))


            theta_dist =  dist.Categorical(torch.eye(self.S * self.K + 1)).expand([data.F, self.S * self.K + 1])  # N, F, state_dim, event_shape
            theta_init = pi_theta_calc(param("pi"), self.K, self.S)  # N, state_dim
            theta_trans = theta_trans_calc(param("A"), self.K, self.S) # N, F, state_dim, state_dim
            theta = pyro.sample("theta", dist.DiscreteHMM(theta_init, theta_trans, theta_dist))

            theta_mask = Vindex(self.theta_matrix)[..., theta]
            m_mask = Vindex(self.m_matrix)[..., theta]

            with K_plate:
                pi_m = pi_m_calc(param("lamda"), self.S)
                m_mask = pyro.sample("m", dist.Categorical(Vindex(pi_m)[m_mask]).to_event(1))
                height = pyro.sample(
                    "height", dist.Gamma(
                        param("height_loc")[m_mask] * param("height_beta")[m_mask],
                        param("height_beta")[m_mask]).to_event(1))
                width = pyro.sample(
                    "width", ScaledBeta(
                        param("width_mode"),
                        param("width_size"), 0.5, 2.5).expand([data.F]).to_event(1))
                x = pyro.sample(
                    "x", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1).to_event(1))
                y = pyro.sample(
                    "y", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+1)/2, data.D+1).to_event(1))

            width = width * 2.5 + 0.5
            x = x * (data.D+1) - (data.D+1)/2
            y = y * (data.D+1) - (data.D+1)/2

            locs = data.loc(height, width, x, y, background, batch_idx)
            pyro.sample(
                "data", self.CameraUnit(
                    locs, param("gain"), param("offset")).to_event(3),
                obs=data[batch_idx])

    def spot_guide(self, data, prefix):
        K_plate = pyro.plate("K_plate", self.K, dim=-2)
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-1)

        with N_plate as batch_idx:
            self.batch_idx = batch_idx.cpu()
            pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/b_loc")[batch_idx]
                    * param(f"{prefix}/b_beta")[batch_idx],
                    param(f"{prefix}/b_beta")[batch_idx]).to_event(1))

            theta = pyro.sample("theta", dist.Categorical(
                param(f"{prefix}/theta_probs")[batch_idx]).to_event(1))
            m_mask = Vindex(self.m_matrix)[..., theta]

            with K_plate:
                pyro.sample("m", dist.Categorical(
                    Vindex(param(f"{prefix}/m_probs")[:, batch_idx])[..., m_mask, :]).to_event(1))

                pyro.sample(
                    "height", dist.Gamma(
                        param(f"{prefix}/h_loc")[:, batch_idx]
                        * param(f"{prefix}/h_beta")[:, batch_idx],
                        param(f"{prefix}/h_beta")[:, batch_idx]).to_event(1))
                pyro.sample(
                    "width", ScaledBeta(
                        param(f"{prefix}/w_mode")[:, batch_idx],
                        param(f"{prefix}/w_size")[:, batch_idx],
                        0.5, 2.5).to_event(1))
                pyro.sample(
                    "x", ScaledBeta(
                        param(f"{prefix}/x_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1).to_event(1))
                pyro.sample(
                    "y", ScaledBeta(
                        param(f"{prefix}/y_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1).to_event(1))

    def spot_parameters(self, data, prefix):
        m_probs = torch.zeros(self.K, data.N, data.F, self.S+1, self.S+1)
        m_probs[..., 0, :] = 1
        for s in range(self.S+1):
            m_probs[..., s, s] = 1
        param(f"{prefix}/m_probs",
              m_probs,
              constraint=constraints.simplex)

        theta_probs = torch.ones(
            data.N, data.F, self.S*self.K+1)
        param(f"{prefix}/theta_probs", theta_probs,
              constraint=constraints.simplex)

    def model_parameters(self, data):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+1)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        param("height_loc", torch.tensor([100., 1000., 2000.])[:self.S+1],
              constraint=constraints.positive)
        param("height_beta", torch.tensor([0.01, 0.01, 0.01])[:self.S+1],
              constraint=constraints.positive)
        param("pi", torch.ones(self.S+1), constraint=constraints.simplex)
        param("lamda", torch.ones(self.S+1), constraint=constraints.simplex)
        param("A", torch.ones(self.S+1, self.S+1), constraint=constraints.simplex)


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
