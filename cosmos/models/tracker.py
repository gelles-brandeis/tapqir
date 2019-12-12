import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine
import pyro.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn
from cosmos.models.helper import Model 

def m_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    m_pi = torch.zeros(2**K)
    m_pi[0] = bernoulli(0.) * poisson(0.)
    k = 1
    for m in range(1,K+1):
        r = int(math.factorial(K)/(math.factorial(K-m)*math.factorial(m)))
        for _ in range(r):
            m_pi[k] = (bernoulli(1) * poisson(m-1) + bernoulli(0) * poisson(m)) / r
            k += 1
    return m_pi

def theta_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    theta_pi = torch.zeros(2**K,K+1)
    theta_pi[0,0] = 1
    theta_pi[1,0] = bernoulli(0) * poisson(1) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[1,1] = bernoulli(1) * poisson(0) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[2,0] = bernoulli(0) * poisson(1) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[2,2] = bernoulli(1) * poisson(0) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[3,0] = bernoulli(0) * poisson(2) / m_param(pi, lamda, K)[3]
    theta_pi[3,1] = bernoulli(1) * poisson(1) / (m_param(pi, lamda, K)[3] * 2)
    theta_pi[3,2] = bernoulli(1) * poisson(1) / (m_param(pi, lamda, K)[3] * 2)
    return theta_pi
    #k = 1
    #for m in range(1,K+1):
    #    r = int(math.factorial(K)/(math.factorial(K-m)*math.factorial(m)))
    #    for _ in range(r):
    #        theta_pi[k,0] = bernoulli(0) * poisson(m) / m_param(pi, lamda, K)[m]
    #        theta_pi[k,1:k+1] = bernoulli(1) * poisson(m-1) / m_param(pi, lamda, K)[m]
    #        k += 1


class Tracker(Model):
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, n_batch, jit, noise="GammaOffset"):
        super().__init__(data, dataset, K, lr, n_batch, jit, noise="GammaOffset")
        self.__name__ = "tracker"
        
        pyro.clear_param_store()
        pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "detector/K{}".format(self.K), "params"))
        self.epoch_count = 0
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "{}".format(self.__name__), "K{}".format(self.K)))
        self.mcc = True 
        
    def model(self):
        # noise variables
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(0., offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(50.))

        #plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        lamda = pyro.sample("lamda", dist.Gamma(0.01 * torch.ones(1), torch.tensor(0.1)))
        m_pi = m_param(pi, lamda, self.K) # 2**K
        theta_pi = theta_param(pi, lamda, self.K) # 2**K,K+1
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]]) # 2**K,K
        theta_matrix = torch.tensor([[0,0], [1,0], [0,1]]) # K+1,K
        height_loc = pyro.sample("height_loc", dist.HalfNormal(200.))
        height_beta = pyro.sample("height_beta", dist.HalfNormal(10.))
        width_mode = pyro.sample("width_mode", self.Location(1.3, 3., 0.5, 2.5))
        width_size = pyro.sample("width_size", dist.HalfNormal(100.))

        # Global Variables
        scale = torch.tensor([10., 0.5])

        #width = pyro.sample("width", Location(torch.tensor(1.3), torch.tensor(10.), 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.HalfNormal(1000.))
                m = pyro.sample("m", dist.Categorical(m_pi)) # N,F,1,1
                theta = pyro.sample("theta", dist.Categorical(theta_pi[m])) # N,F,1,1
                m = m_matrix[m] # N,F,1,1,K
                theta = theta_matrix[theta] # N,F,1,1,K   K+1,1,1,1,1,1
                height = pyro.sample("height", dist.Gamma(height_loc * height_beta, height_beta).expand([len(batch_idx),self.F,1,1,self.K]).mask(m).to_event(1)) # K,N,F,1,1
                height = height.masked_fill(~m.bool(), 0.)
                width = pyro.sample("width", self.Location(width_mode, width_size, 0.5, 2.5).expand([len(batch_idx),self.F,1,1,self.K]).mask(m).to_event(1))
                x0 = pyro.sample("x0", dist.Normal(0., scale[theta]).mask(m).to_event(1))
                y0 = pyro.sample("y0", dist.Normal(0., scale[theta]).mask(m).to_event(1))

                locs = self.gaussian_spot(batch_idx, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, gain, offset), obs=self.data[batch_idx])
    
    def guide(self):
        offset_max = self.data._store.min() - 0.1
        offset_v = pyro.param("offset_v", offset_max-50, constraint=constraints.interval(0.,offset_max))
        gain_v = pyro.param("gain_v", torch.tensor(5.), constraint=constraints.positive)
        pyro.sample("offset", dist.Delta(offset_v))
        pyro.sample("gain", dist.Delta(gain_v))

        # plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)

        # Global Parameters
        b_loc, b_beta = pyro.param("b_loc"), pyro.param("b_beta")
        w_mode, w_size= pyro.param("w_mode"), pyro.param("w_size")
        h_loc, h_beta = pyro.param("h_loc"), pyro.param("h_beta")
        x_mean, y_mean, scale = pyro.param("x_mean"), pyro.param("y_mean"), pyro.param("scale")

        m_probs = pyro.param("m_probs")
        height_loc_v = pyro.param("height_loc_v")
        height_beta_v = pyro.param("height_beta_v")
        width_mode_v = pyro.param("width_mode_v")
        width_size_v = pyro.param("width_size_v")

        ######
        theta_probs = pyro.param("theta_probs", torch.ones(self.N,self.F,1,1,self.K+1), constraint=constraints.simplex)
        pi_concentration = pyro.param("pi_concentration", torch.ones(2)*self.N*self.F/2, constraint=constraints.positive)
        lamda_loc = pyro.param("lamda_loc", torch.tensor([0.1]), constraint=constraints.positive)
        lamda_beta = pyro.param("lamda_beta", torch.tensor([10.]), constraint=constraints.positive)
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]]) # 2**K,K


        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("lamda", dist.Gamma(lamda_loc * lamda_beta, lamda_beta))
        pyro.sample("height_loc", dist.Delta(height_loc_v))
        pyro.sample("height_beta", dist.Delta(height_beta_v))
        pyro.sample("width_mode", dist.Delta(width_mode_v))
        pyro.sample("width_size", dist.Delta(width_size_v))
        #width = pyro.sample("width", Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.Gamma(b_loc[batch_idx] * self.D**2, self.D**2))
                m = pyro.sample("m", dist.Categorical(m_probs[batch_idx]), infer={"enumerate": "sequential"})
                m = m_matrix[m] # N,F,1,1,K
                pyro.sample("theta", dist.Categorical(theta_probs[batch_idx]), infer={"enumerate": "parallel"}) # N,F,1,1
                pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta, h_beta).mask(m).to_event(1))
                pyro.sample("width", self.Location(w_mode[batch_idx], w_size[batch_idx], 0.5, 2.5).mask(m).to_event(1))
                pyro.sample("x0", dist.Normal(x_mean[batch_idx], scale[batch_idx]).mask(m).to_event(1))
                pyro.sample("y0", dist.Normal(y_mean[batch_idx], scale[batch_idx]).mask(m).to_event(1))


