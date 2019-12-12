import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
from pyro.infer import config_enumerate
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


class Features(Model):
    """ Extract features of the Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, n_batch, jit, noise="GammaOffset"):
        super().__init__(data, dataset, K, lr, n_batch, jit, noise="GammaOffset")
        self.__name__ = "features"
        
        pyro.clear_param_store()
        self.epoch_count = 0
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = JitTrace_ELBO() if jit else Trace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "{}".format(self.__name__), "K{}".format(self.K)))
        self.mcc = False
    
    def model(self):
        # noise variables
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(0., offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(50.))

        #plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        with N_plate as batch_idx:
            with F_plate:
                background = pyro.sample("background", dist.HalfNormal(1000.))
                height = pyro.sample("height", dist.HalfNormal(3000.).expand([1,1,1,1,self.K]).to_event(1)) # N,F,1,1,K
                width = pyro.sample("width", self.Location(1.3, 10., 0.5, 2.5).expand([1,1,1,1,self.K]).to_event(1))
                x0 = pyro.sample("x0", dist.Normal(torch.tensor(0.), 10.).expand([1,1,1,1,self.K]).to_event(1))
                y0 = pyro.sample("y0", dist.Normal(torch.tensor(0.), 10.).expand([1,1,1,1,self.K]).to_event(1))

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
        b_loc = pyro.param("b_loc", torch.ones(self.N,self.F,1,1)*10., constraint=constraints.positive)
        b_beta = pyro.param("b_beta", torch.ones(1)*self.D**2, constraint=constraints.positive)
        w_mode = pyro.param("w_mode", torch.ones(self.N,self.F,1,1,self.K)*1.35, constraint=constraints.interval(0.5,3.))
        w_size = pyro.param("w_size", torch.ones(self.N,self.F,1,1,self.K)*100., constraint=constraints.greater_than(2.))
        #w_mode = pyro.param("w_mode", torch.ones(1)*1.35, constraint=constraints.interval(0.5,3.))
        #w_size = pyro.param("w_size", torch.ones(1)*1000., constraint=constraints.greater_than(2.))
        intensity = torch.ones(self.N,self.F,1,1,self.K)*10.
        intensity[...,1] = 30. # 30 is better than 100
        h_loc = pyro.param("h_loc", intensity, constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(1), constraint=constraints.positive)
        x_mean = pyro.param("x_mean", torch.zeros(self.N,self.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        y_mean = pyro.param("y_mean", torch.zeros(self.N,self.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #var = (w_mode**2 + 1/12) / h_loc + 8 * math.pi * w_mode**4 * b_loc.unsqueeze(dim=-1) / h_loc**2
        #size = ((self.D+3)**2/4) / var - 1
        #x_conc0 = pyro.param("x_conc0", size/2, constraint=constraints.greater_than(1.))
        #x_conc1 = pyro.param("x_conc1", size/2, constraint=constraints.greater_than(1.))
        #y_conc0 = pyro.param("y_conc0", size/2, constraint=constraints.greater_than(1.))
        #y_conc1 = pyro.param("y_conc1", size/2, constraint=constraints.greater_than(1.))
        scale = torch.sqrt((w_mode**2 + 1/12) / h_loc + 8 * math.pi * w_mode**4 * b_loc.unsqueeze(dim=-1) / h_loc**2)
        scale = pyro.param("scale", scale, constraint=constraints.positive)
        

        #width = pyro.sample("width", Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                #background = pyro.sample("background", dist.Gamma(b_loc[batch_idx] * self.D**2, self.D**2))
                #height = pyro.sample("height", dist.Gamma(h_loc[batch_idx], 1.).to_event(1))
                background = pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta))
                height = pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta, h_beta).to_event(1))
                width = pyro.sample("width", self.Location(w_mode[batch_idx], w_size[batch_idx], 0.5, 2.5).to_event(1))
                #scale = torch.sqrt((width**2 + 1/12) / height + 8 * math.pi * width**4 * background.unsqueeze(dim=-1) / height**2)
                pyro.sample("x0", dist.Normal(x_mean[batch_idx], scale[batch_idx]).to_event(1))
                pyro.sample("y0", dist.Normal(y_mean[batch_idx], scale[batch_idx]).to_event(1))
                #pyro.sample("x0", dist.Normal(x_mode[batch_idx], scale).to_event(1))
                #pyro.sample("y0", dist.Normal(y_mode[batch_idx], scale).to_event(1))
                #var = (width**2 + 1/12) / height + 8 * math.pi * width**4 * background.unsqueeze(dim=-1) / height**2
                #x_size = ((self.D+3)**2/4 - x_mode[batch_idx]**2) / var - 1
                #y_size = ((self.D+3)**2/4 - y_mode[batch_idx]**2) / var - 1
                #x_size = torch.where(x_size > 2., x_size, torch.tensor([2.]))
                #y_size = torch.where(y_size > 2., y_size, torch.tensor([2.]))
                #pyro.sample("x0", self.Location2(x_conc0[batch_idx], x_conc1[batch_idx], -(self.D+3)/2, self.D+3).to_event(1))
                #pyro.sample("y0", self.Location2(y_conc0[batch_idx], y_conc1[batch_idx], -(self.D+3)/2, self.D+3).to_event(1))
