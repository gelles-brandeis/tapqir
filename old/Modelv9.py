import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.constraints import Constraint
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
from cosmos.models.helper import Location

class m_probs_constraint(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """
    def check(self, value):
        return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6) & (value[...,1].sum(0)  <= 1.)

def per_param_args(module_name, param_name):
    if param_name in ["size", "jsize"]:
        return {"lr": 0.005}
    else:
        return {"lr": 0.002}

class Modelv9:
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, n_batch, jit, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "v9"
        self.data = data
        self.dataset = dataset
        self.N, self.F, self.D, _ = data._store.shape
        assert K >= 2
        self.K = K
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        self.h_loc = 10 
        self.lr = lr
        self.n_batch = n_batch
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).to(torch.float32)
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(1,self.N,self.F,1,1,2).repeat(self.K,1,1,1,1,1)# N,F,1,1,M,K,2
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)
        
        pyro.clear_param_store()
        pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "detector", "v10/M2", "params"))
        self.epoch_count = 0
        #self.prefit_optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        #self.optim = pyro.optim.Adam(per_param_args)
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["height_loc_v", "height_beta_v"]), 
                                self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "detector", "{}".format(self.__name__), "M{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = self.target_locs[:,batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
        spot_locs[...,0] += x0 # N,F,D,D,M,K,2 .permute(1,2,3,4,5,0) # adjust for the center of the first frame
        spot_locs[...,1] += y0 #.permute(1,2,3,4,5,0) # x0 and y0 can be either scalars or vectors
        # N,F,D,D,M,K -> tril
        #height[...,1,:] = 0
        #spot = torch.zeros(len(batch_idx),self.F,self.D,self.D,self.K+1)
        #height = height[...,self.M-1,:self.M]
        spot = torch.zeros(batch_idx.shape[0],self.F,self.D,self.D)
        for k in range(self.K):
            #spot_locs = spot_locs.reshape(len(batch_idx),self.F,1,1,2) 
            #w = width.reshape(1,1,1,1)
            w = width[k]
            rv = dist.MultivariateNormal(spot_locs[k], scale_tril=self.spot_scale * w.view(w.size()+(1,1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D,M
            spot += height[k] * gaussian_spot # N,F,D,D,M
        return spot
    
    @config_enumerate
    def model(self):
        # noise variables
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(torch.tensor(0.), offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(torch.tensor(50.)))

        #plates
        K_plate = pyro.plate("K_plate", self.K, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        # Global Variables
        #with K_plate:
        m_pi = pyro.sample("m_pi", dist.Dirichlet(torch.ones(2) / 2))
        #m_pi = pyro.sample("m_pi", dist.Dirichlet(torch.ones(3) / 3))
        height_loc = pyro.sample("height_loc", dist.HalfNormal(torch.tensor([50., 3000.])).to_event(1))
        height_beta = pyro.sample("height_beta", dist.HalfNormal(torch.tensor([10., 10.])).to_event(1))
        #height_loc = torch.cat([height_loc, height_loc[1:]])
        #height_beta = torch.cat([height_beta, height_beta[1:]])
        #background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.))
        #background_beta = pyro.sample("background_beta", dist.HalfNormal(100.))
        
        #with K_plate:
        #width_mode = pyro.sample("width_mode", Location(1.3, 4., 0.5, 2.5))
        #width_size = pyro.sample("width_size", dist.HalfNormal(500.))
                
        #x0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1), 2.])
        #y0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1), 2.])
        x0_size = torch.tensor([2., 2.])
        y0_size = torch.tensor([2., 2.])
        #x0_size = 2. * torch.ones(1)
        #y0_size = 2. * torch.ones(1)

        #width = pyro.sample("width", Location(torch.tensor(1.3), torch.tensor(10.), 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                background = pyro.sample("background", dist.HalfNormal(1000.))
                with K_plate:
                    m = pyro.sample("m", dist.Categorical(m_pi)) # K,N,F,1,1
                    # AoI & Frame Local Variables
                    #background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta).expand([1,1,1,1]).to_event(2))
                    height = pyro.sample("height", dist.Gamma(height_loc[m] * height_beta[m], height_beta[m])) # K,N,F,1,1
                    width = pyro.sample("width", Location(1.3, 10., 0.5, 2.5))
                    x0 = pyro.sample("x0", Location(0., x0_size[m], -(self.D+3)/2, self.D+3))
                    y0 = pyro.sample("y0", Location(0., y0_size[m], -(self.D+3)/2, self.D+3))

                locs = self.gaussian_spot(batch_idx, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, gain, offset), obs=self.data[batch_idx])
    
    @config_enumerate
    def guide(self):
        offset_max = self.data._store.min() - 0.1
        offset_v = pyro.param("offset_v", offset_max-50, constraint=constraints.interval(0,offset_max.item()))
        gain_v = pyro.param("gain_v", torch.tensor(5.), constraint=constraints.positive)
        pyro.sample("offset", dist.Delta(offset_v))
        pyro.sample("gain", dist.Delta(gain_v))


        # plates
        K_plate = pyro.plate("K_plate", self.K, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)


        b_loc, b_beta= pyro.param("b_loc"), pyro.param("b_beta")
        w_mode, w_size= pyro.param("w_mode"), pyro.param("w_size")
        h_loc, h_beta= pyro.param("h_loc"), pyro.param("h_beta")
        x_mode, y_mode, size= pyro.param("x_mode"), pyro.param("y_mode"), pyro.param("size")

        height_loc_v = pyro.param("height_loc_v", torch.tensor([10., 1500.]), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.tensor([0.1, 0.01]), constraint=constraints.positive)

        ######
        h_max = np.percentile(pyro.param("h_loc").detach().cpu(), 95)
        p = torch.where(pyro.param("h_loc").detach() < h_max, pyro.param("h_loc").detach()/h_max, torch.tensor(1.))

        #m2 = p * 0.45 + 0.025
        #m1 = p * 0.45 + 0.025
        #m0 = 1 - m1 - m2
        m1 = p * 0.9 + 0.05
        m0 = 1 - m1

        m_probs = torch.stack((m0,m1), dim=-1)
        m_probs = pyro.param("m_probs", m_probs.reshape(self.K,self.N,self.F,1,1,2), constraint=constraints.simplex)
        #m_probs = torch.stack((m0,m1,m2), dim=-1)
        #m_probs = pyro.param("m_probs", m_probs.reshape(self.K,self.N,self.F,1,1,3), constraint=constraints.simplex)
        #m_probs = m_probs.clamp(min=-40, max=40)
        
        # Global Parameters
        #m_pi_concentration = pyro.param("m_pi_concentration", m_probs.sum(dim=(0,1,2,3)), constraint=constraints.positive)
        #m_pi_concentration = pyro.param("m_pi_concentration", torch.ones(2)*self.N*self.F/2, constraint=constraints.positive)
        m_pi_concentration = pyro.param("m_pi_concentration", torch.ones(2)*1000, constraint=constraints.positive)

        # Global Variables
        #with K_plate:
        pyro.sample("m_pi", dist.Dirichlet(m_pi_concentration))
        pyro.sample("height_loc", dist.Delta(height_loc_v).to_event(1))
        pyro.sample("height_beta", dist.Delta(height_beta_v).to_event(1))

        #pyro.sample("width", Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta))
                with K_plate:
                    m = pyro.sample("m", dist.Categorical(m_probs[:,batch_idx]))
                    # AoI & Frame Local Variables
                    height = pyro.sample("height", dist.Gamma(h_loc[:,batch_idx] * h_beta[:,batch_idx], h_beta[:,batch_idx]))
                    #pyro.sample("height", dist.Gamma(h_loc * size[:,batch_idx] * h_beta[:,batch_idx], h_beta[:,batch_idx]))
                    pyro.sample("width", Location(w_mode[:,batch_idx], w_size[:,batch_idx], 0.5, 2.5))
                    pyro.sample("x0", Location(x_mode[:,batch_idx], (size * height + 2.), -(self.D+3)/2, self.D+3))
                    pyro.sample("y0", Location(y_mode[:,batch_idx], (size * height + 2.), -(self.D+3)/2, self.D+3))


    def fixed_epoch(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.fixed.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.fixed, self.writer, feature=False)
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()


    def epoch(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.svi.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False)
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        self.optim.save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "M{}".format(self.K), "optimizer"))
        pyro.get_param_store().save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "M{}".format(self.K), "params"))
        np.savetxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "M{}".format(self.K), "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("Classification results were saved in {}...".format(self.data.path))

    def load(self):
        try:
            self.epoch_count = int(np.loadtxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                    "{}".format(self.__name__), "M{}".format(self.K), "epoch_count")))
            self.optim.load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                    "{}".format(self.__name__), "M{}".format(self.K), "optimizer"))
            pyro.get_param_store().load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                    "{}".format(self.__name__), "M{}".format(self.K), "params"))
            print("loaded previous run")
        except:
            pass
