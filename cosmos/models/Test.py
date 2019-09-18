import math
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine
import pyro.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn

class Test:
    """ Gaussian Spot Model """
    def __init__(self, data, K, lr, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "test"
        self.data = data
        self.N, self.F, self.D = 331, 2366, 10 
        assert K >= 2
        self.K = K
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).float()
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)
        
        pyro.clear_param_store()
        self.epoch_count = 0
        self.lr = lr
        #self.prefit_optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["h_loc", "h_beta"
            "w_loc", "w_beta", "height_loc_v"]), self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo) 
        #self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "classifier", "{}".format(self.__name__), "K{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width):
        rv = dist.MultivariateNormal(torch.tensor([4.5]), scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        return height * gaussian_spot #
    
    @config_enumerate
    def model(self, batch_idx):
        # noise variables
        noise_params = dict()
        for var in self._params:
            noise_params[var] = pyro.sample(var, self._params[var]["prior"])

        #plates
        K_plate = pyro.plate("K_plate", self.K-1) 
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(10.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))

        height_loc = torch.cat((torch.ones(1), height_loc), 0)
        height_beta = torch.cat((torch.ones(1), height_beta), 0)
        width_loc = torch.cat((torch.ones(1), width_loc), 0)
        width_beta = torch.cat((torch.ones(1), width_beta), 0)
                

        with N_plate:
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(pi))
                with poutine.mask(mask=(z > 0).byte()):
                    height = pyro.sample("height", dist.Gamma(height_loc[z] * height_beta[z], height_beta[z]))
                    width = pyro.sample("width", dist.Gamma(width_loc[z] * width_beta[z], width_beta[z]))
            
                # return locs for K classes
                spots = self.gaussian_spot(batch_idx, height, width)
                locs = torch.where(z > 0, spots, torch.zeros_like(spots)) + 250 
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, **noise_params), obs=self.data[batch_idx])
        return locs
    
    @config_enumerate
    def guide(self, batch_idx):
        # noise variables
        for var in self._params:
            guide_params = dict()
            for param in self._params[var]["guide_params"]:
                guide_params[param] = pyro.param(**self._params[var]["guide_params"][param]) 
            pyro.sample(var, self._params[var]["guide_dist"](**guide_params))

        # plates
        K_plate = pyro.plate("K_plate", self.K-1)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", 100*torch.ones(self.K-1), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K-1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", self.data.w_loc.mean()*torch.ones(self.K-1), constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1), constraint=constraints.positive)

        # AoI & Frame Local Parameters
        h_loc = pyro.param("h_loc", self.data.h_loc.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", self.data.h_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        w_loc = pyro.param("w_loc", self.data.w_loc.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        w_beta = pyro.param("w_beta", self.data.w_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)

        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        with K_plate:
                pyro.sample("height_loc", dist.Delta(height_loc_v))
                pyro.sample("height_beta", dist.Delta(height_beta_v))
                pyro.sample("width_loc", dist.Delta(width_loc_v))
                pyro.sample("width_beta", dist.Delta(width_beta_v))
        
        with N_plate:
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(z_probs[batch_idx]))
                with poutine.mask(mask=(z > 0).byte()):
                    pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta[batch_idx], h_beta[batch_idx]))
                    pyro.sample("width", dist.Gamma(w_loc[batch_idx] * w_beta[batch_idx], w_beta[batch_idx]))
        

    def fixed_epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                loss = self.fixed.step(batch_idx)
                epoch_loss += loss * len(batch_idx) / self.N
            if not (self.epoch_count % 10):    
                write_summary(self.epoch_count, epoch_loss, self, self.fixed, self.writer, feature=False, mcc=False)
            self.epoch_count += 1

    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                loss = self.svi.step(batch_idx)
                epoch_loss += loss * len(batch_idx) / self.N
            if not (self.epoch_count % 10):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False, mcc=False)
            self.epoch_count += 1
