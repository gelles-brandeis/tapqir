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
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
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


class Detector(Model):
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, n_batch, jit, noise="GammaOffset"):
        super().__init__(data, dataset, K, lr, n_batch, jit, noise="GammaOffset")
        self.__name__ = "detector"
        
        pyro.clear_param_store()
        pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "features/K{}".format(self.K), "params"))
        self.epoch_count = 0
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        #self.elbo = JitTrace_ELBO() if jit else Trace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "{}".format(self.__name__), "K{}".format(self.K)))
        self.mcc = False
    
    #@config_enumerate
    def model(self):
        # noise variables
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(0., offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(50.))

        #plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        m_pi = pyro.sample("m_pi", dist.Dirichlet(torch.ones(4) / 4))
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])
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
                m = m_matrix[m] # N,F,1,1,K
                height = pyro.sample("height", dist.Gamma(height_loc * height_beta, height_beta).expand([len(batch_idx),self.F,1,1,self.K]).mask(m).to_event(1)) # K,N,F,1,1
                height = height.masked_fill(~m.bool(), 0.)
                #height = pyro.sample("height", dist.Gamma(height_loc[m] * height_beta[m], height_beta[m]).to_event(1)) # K,N,F,1,1
                width = pyro.sample("width", self.Location(width_mode, width_size, 0.5, 2.5).expand([len(batch_idx),self.F,1,1,self.K]).mask(m).to_event(1))
                #width = pyro.sample("width", self.Location(1.3, 10., 0.5, 2.5).expand([1,1,1,1,self.K]).mask(m).to_event(1))
                x0 = pyro.sample("x0", dist.Normal(torch.tensor(0.), 10.).expand([len(batch_idx),self.F,1,1,self.K]).mask(m).to_event(1))
                y0 = pyro.sample("y0", dist.Normal(torch.tensor(0.), 10.).expand([len(batch_idx),self.F,1,1,self.K]).mask(m).to_event(1))

                locs = self.gaussian_spot(batch_idx, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, gain, offset), obs=self.data[batch_idx])
    
    #@config_enumerate
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
        
        height_loc_v = pyro.param("height_loc_v", torch.tensor([200.]), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.tensor([1.]), constraint=constraints.positive)
        width_mode_v = pyro.param("width_mode_v", torch.tensor([1.3]), constraint=constraints.positive)
        width_size_v = pyro.param("width_size_v", torch.tensor([100.]), constraint=constraints.positive)

        ######
        h_max = np.percentile(pyro.param("h_loc").detach().cpu(), 95)
        p = torch.where(pyro.param("h_loc").detach() < h_max, pyro.param("h_loc").detach()/h_max, torch.tensor(1.))

        m3 = 0.9 * p[...,0] * p[...,1] + 0.025
        m2 = 0.9 * (1 - p[...,0]) * p[...,1] + 0.025
        m1 = 0.9 * p[...,0] * (1 - p[...,1]) + 0.025
        m0 = 0.9 * (1 - p[...,0]) * (1 - p[...,1]) + 0.025
        #m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])

        m_probs = torch.stack((m0,m1,m2,m3), dim=-1)
        m_probs = pyro.param("m_probs", m_probs.reshape(self.N,self.F,1,1,4), constraint=constraints.simplex)
        #m_probs = pyro.param("m_probs", torch.ones(self.N,self.F,1,1,4), constraint=constraints.simplex)
        m_pi_concentration = pyro.param("m_pi_concentration", torch.ones(4)*self.N*self.F/4, constraint=constraints.positive)
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])

        pyro.sample("m_pi", dist.Dirichlet(m_pi_concentration))
        pyro.sample("height_loc", dist.Delta(height_loc_v))
        pyro.sample("height_beta", dist.Delta(height_beta_v))
        pyro.sample("width_mode", dist.Delta(width_mode_v))
        pyro.sample("width_size", dist.Delta(width_size_v))
        #width = pyro.sample("width", Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta))
                m = pyro.sample("m", dist.Categorical(m_probs[batch_idx]), infer={"enumerate": "sequential"})
                m = m_matrix[m] # N,F,1,1,K
                height = pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta, h_beta).mask(m).to_event(1))
                width = pyro.sample("width", self.Location(w_mode[batch_idx], w_size[batch_idx], 0.5, 2.5).mask(m).to_event(1))
                #scale = torch.sqrt((width**2 + 1/12) / height + 8 * math.pi * width**4 * background.unsqueeze(dim=-1) / height**2)
                pyro.sample("x0", dist.Normal(x_mean[batch_idx], scale[batch_idx]).mask(m).to_event(1))
                pyro.sample("y0", dist.Normal(y_mean[batch_idx], scale[batch_idx]).mask(m).to_event(1))
                #pyro.sample("x0", Location(x_mode[:,batch_idx], size[:,batch_idx], -(self.D+3)/2, self.D+3))
                #pyro.sample("y0", Location(y_mode[:,batch_idx], size[:,batch_idx], -(self.D+3)/2, self.D+3))

