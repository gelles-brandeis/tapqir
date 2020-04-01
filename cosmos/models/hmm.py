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
from torch.distributions.utils import probs_to_logits
from pyro.distributions import TorchDistribution

from cosmos.models.model import Model
from cosmos.models.helper import pi_m_calc, pi_theta_calc, theta_trans_calc, trans_calc, init_calc


class StackDistributions(TorchDistribution):
    """
    Stack multiple heterogeneous distributions.

    This is useful when multiple heterogeneous distributions
    depend on the same hidden state in DiscreteHMM.

    Example::

        d1 = dist.Normal(torch.zeros(3), 1.)
        d2 = dist.Gamma(torch.ones(3), 1.)
        d = StackDistributions(d1, d2)

    :param sequence of pyro.distributions.TorchDistribution distributions
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
        batch_shape = emission_logits.shape[:-1]
        event_shape = self.dist.event_shape
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        #value = value.unsqueeze(-1 - self.dist.event_dim)
        obs_logits = self.dist.log_prob(value)
        result = obs_logits.unsqueeze(dim=-2) + self.emission_logits
        result = torch.logsumexp(result, -1)
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

        #self.spot_model(self.data, prefix="d")
        self.viterbi_model(self.data)

        if self.control:
            self.spot_model(self.control, prefix="c")

    def guide(self):
        self.guide_parameters()
        #self.spot_guide(self.data, prefix="d")
        self.viterbi_guide(self.data)

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


            with K_plate:
                pi_m = pi_m_calc(param("lamda"), self.S)
                m_logits = Vindex(pi_m)[self.m_matrix].log()
                h_dist = EnumDistribution(dist.Gamma(
                        param("height_loc") * param("height_beta"),
                        param("height_beta")), m_logits)
                x_dist = ScaledBeta(
                        0, self.size[self.theta_matrix], -(data.D+1)/2, data.D+1)
                y_dist = ScaledBeta(
                        0, self.size[self.theta_matrix], -(data.D+1)/2, data.D+1)
                #h_dist = dist.Gamma(
                #        param("height_loc")[self.m_state] * param("height_beta")[self.m_state],
                #        param("height_beta")[self.m_state])
                #x_dist = ScaledBeta(
                #        0, self.size[self.theta_state], -(data.D+1)/2, data.D+1)
                #y_dist = ScaledBeta(
                #        0, self.size[self.theta_state], -(data.D+1)/2, data.D+1)
                hxy_dist = StackDistributions(h_dist, x_dist, y_dist)

                init = pi_theta_calc(param("pi"), self.K, self.S).log()  # N, state_dim
                trans = theta_trans_calc(param("A"), self.K, self.S).log() # N, F, state_dim, state_dim
                #init = init_calc(param("pi"), param("lamda"))  # N, state_dim
                #trans = trans_calc(param("A"), param("lamda")) # N, F, state_dim, state_dim
                hmm_dist = dist.DiscreteHMM(init, trans, hxy_dist, duration=data.F)

                hxy = pyro.sample("hxy", hmm_dist)
                height, x, y = torch.unbind(hxy, dim=-1)

                width = pyro.sample(
                    "width", ScaledBeta(
                        param("width_mode"),
                        param("width_size"), 0.5, 2.5).expand([data.F]).to_event(1))

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

            with K_plate:
                h_dist = dist.Gamma(
                        param(f"{prefix}/h_loc")[:, batch_idx]
                        * param(f"{prefix}/h_beta")[:, batch_idx],
                        param(f"{prefix}/h_beta")[:, batch_idx]).to_event(1)
                x_dist = ScaledBeta(
                        param(f"{prefix}/x_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1).to_event(1)
                y_dist = ScaledBeta(
                        param(f"{prefix}/y_mode")[:, batch_idx],
                        param(f"{prefix}/size")[:, batch_idx],
                        -(data.D+1)/2, data.D+1).to_event(1)
                hxy_dist = StackDistributions(h_dist, x_dist, y_dist)
                pyro.sample(
                    "hxy", hxy_dist)
                pyro.sample(
                    "width", ScaledBeta(
                        param(f"{prefix}/w_mode")[:, batch_idx],
                        param(f"{prefix}/w_size")[:, batch_idx],
                        0.5, 2.5).to_event(1))

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
    def viterbi_model(self, data):
        K_plate = pyro.plate("K_plate", self.K, dim=-2)
        N_plate = pyro.plate("N_plate", data.N, dim=-1)
        init = pi_theta_calc(param("pi"), self.K, self.S)  # N, state_dim
        trans = theta_trans_calc(param("A"), self.K, self.S) # N, F, state_dim, state_dim
        with N_plate as batch_idx:

            init = pi_theta_calc(param("pi"), self.K, self.S)  # N, state_dim
            trans = theta_trans_calc(param("A"), self.K, self.S) # N, F, state_dim, state_dim
            pi_m = pi_m_calc(param("lamda"), self.S)
            theta = pyro.sample("theta", dist.Categorical(init))
            thetas = []
            for f in pyro.markov(range(data.F)):
                background = pyro.sample(
                    f"background_{f}", dist.Gamma(
                        param(f"d/background_loc")[batch_idx, 0]
                        * param("background_beta"), param("background_beta")))

                theta = pyro.sample(
                    f"theta_{f}", dist.Categorical(Vindex(trans)[theta, :]))
                theta_mask = Vindex(self.theta_matrix)[..., 0, theta]
                m_mask = Vindex(self.m_matrix)[..., 0, theta]

                with K_plate:
                    m_mask = pyro.sample(f"m_{f}", dist.Categorical(Vindex(pi_m)[m_mask]))
                    height = pyro.sample(
                        f"height_{f}", dist.Gamma(
                            param("height_loc")[m_mask] * param("height_beta")[m_mask],
                            param("height_beta")[m_mask]))
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

                locs = data.loc(height, width, x, y, background, batch_idx, None, f)
                pyro.sample(
                    f"data_{f}", self.CameraUnit(
                        locs, param("gain"), param("offset")).to_event(2),
                    obs=data[batch_idx, f])
                thetas.append(theta)
        return thetas


    def viterbi_guide(self, data):
        K_plate = pyro.plate("K_plate", self.K, dim=-2)
        #N_plate = pyro.plate("N_plate", data.N, dim=-1)
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-1)

        with N_plate as batch_idx:
            pyro.sample(
                "background", dist.Gamma(
                    param("d/b_loc")[batch_idx]
                    * param("d/b_beta")[batch_idx],
                    param("d/b_beta")[batch_idx]).to_event(1))
            with K_plate:
                pyro.sample(
                    "width", ScaledBeta(
                        param("d/w_mode")[:, batch_idx],
                        param("d/w_size")[:, batch_idx],
                        0.5, 2.5).to_event(1))

            for f in pyro.markov(range(data.F)):
                pyro.sample(
                    f"background_{f}", dist.Gamma(
                        param("d/b_loc")[batch_idx, f]
                        * param("d/b_beta")[batch_idx, f],
                        param("d/b_beta")[batch_idx, f]))

                with K_plate:
                    pyro.sample(
                        f"height_{f}", dist.Gamma(
                            param("d/h_loc")[:, batch_idx, f]
                            * param("d/h_beta")[:, batch_idx, f],
                            param("d/h_beta")[:, batch_idx, f]))
                    pyro.sample(
                        f"width_{f}", ScaledBeta(
                            param("d/w_mode")[:, batch_idx, f],
                            param("d/w_size")[:, batch_idx, f],
                            0.5, 2.5))
                    pyro.sample(
                        f"x_{f}", ScaledBeta(
                            param("d/x_mode")[:, batch_idx, f],
                            param("d/size")[:, batch_idx, f],
                            -(data.D+1)/2, data.D+1))
                    pyro.sample(
                        f"y_{f}", ScaledBeta(
                            param("d/y_mode")[:, batch_idx, f],
                            param("d/size")[:, batch_idx, f],
                            -(data.D+1)/2, data.D+1))
