import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.infer import SVI, TraceEnum_ELBO, JitTraceEnum_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine
import pyro.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pyro.util import ignore_jit_warnings

from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn

class Modelv4:
    """ Gaussian Spot Model """
    def __init__(self, data, K, lr, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "v4"
        self.data = data
        self.N, self.F, self.D, _ = data._store.shape
        assert K >= 2
        self.K = K
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).float()
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(self.N,self.F,1,1,2)
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)
        
        pyro.clear_param_store()
        self.epoch_count = 0
        self.lr = lr
        #self.prefit_optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.optim = pyro.optim.AdamW({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["h_loc", "h_beta", "b_loc", "b_beta",
            "w_loc", "w_beta", "x_mode", "x_size", "y_mode", "y_size", "x0_size_v", "y0_size_v", "height_loc_v", 
            "jh_loc", "jh_beta", "jb_loc", "jb_beta",
            "jw_loc", "jw_beta", "jx_mode", "jx_size", "jy_mode", "jy_size"]), self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, poutine.block(self.guide, hide=["x0_size_v", "y0_size_v"]), self.optim, loss=self.elbo) 
        #self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "classifier", "{}".format(self.__name__), "K{}".format(self.K)))
        
    def Location(self, mode, size, loc, scale):
        mode = (mode - loc) / scale
        concentration1 = mode * (size - 2) + 1
        concentration0 = (1 - mode) * (size - 2) + 1
        base_distribution = dist.Beta(concentration1, concentration0)
        transforms =  [AffineTransform(loc=loc, scale=scale)]
        return dist.TransformedDistribution(base_distribution, transforms)

    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        # select target locs for given indices
        spot_locs = self.target_locs[batch_idx] # ind,F,D,D,2
        # adjust for the center of the first frame
        # x0 and y0 can be either scalars or vectors
        spot_locs[...,0] += x0
        spot_locs[...,1] += y0
        rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        # height can be either a scalar or a vector
        return height * gaussian_spot #
    
    @config_enumerate
    def model(self):
        # noise variables
        noise_params = dict()
        for var in self._params:
            noise_params[var] = pyro.sample(var, self._params[var]["prior"])

        #plates
        K_plate = pyro.plate("K_plate", self.K-1) 
        N_plate = pyro.plate("N_plate", self.N, 16, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        junk_pi = pyro.sample("junk_pi", dist.Dirichlet(0.5 * torch.ones(2)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1,1]))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(10.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_size = pyro.sample("x0_size", dist.HalfNormal(1000.))
            y0_size = pyro.sample("y0_size", dist.HalfNormal(1000.))

        height_loc = torch.cat((torch.ones(1), height_loc), 0)
        height_beta = torch.cat((torch.ones(1), height_beta), 0)
        width_loc = torch.cat((torch.ones(1), width_loc), 0)
        width_beta = torch.cat((torch.ones(1), width_beta), 0)
        x0_size = torch.cat((torch.ones(1), x0_size), 0)
        y0_size = torch.cat((torch.ones(1), y0_size), 0)
        x0_size += 2.
        y0_size += 2.
                
        # junk molecules
        #junk_height_loc = pyro.sample("junk_height_loc", dist.HalfNormal(500.))
        #junk_height_beta = pyro.sample("junk_height_beta", dist.HalfNormal(50.))
        #junk_width_loc = pyro.sample("junk_width_loc", dist.HalfNormal(10.))
        #junk_width_beta = pyro.sample("junk_width_beta", dist.HalfNormal(5.))
        
        #junk_height_loc = torch.cat((torch.ones(1), junk_height_loc), 0)
        #junk_height_beta = torch.cat((torch.ones(1), junk_height_beta), 0)
        #junk_width_loc = torch.cat((torch.ones(1), junk_width_loc), 0)
        #junk_width_beta = torch.cat((torch.ones(1), junk_width_beta), 0)
        junk_x0_size = 2. * torch.ones(2)
        junk_y0_size = 2. * torch.ones(2)

        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(pi))
                j = pyro.sample("j", dist.Categorical(junk_pi))
                background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta))
                with poutine.mask(mask=(z > 0).byte()):
                    height = pyro.sample("height", dist.Gamma(height_loc[z] * height_beta[z], height_beta[z]))
                    width = pyro.sample("width", dist.Gamma(width_loc[z] * width_beta[z], width_beta[z]))
                    x0 = pyro.sample("x0", self.Location(0., x0_size[z], -(self.D+3)/2, self.D+3))
                    y0 = pyro.sample("y0", self.Location(0., y0_size[z], -(self.D+3)/2, self.D+3))
                    #x0 = pyro.sample("x0", dist.Normal(0., x0_scale[z]))
                    #y0 = pyro.sample("y0", dist.Normal(0., y0_scale[z]))
                with poutine.mask(mask=j.byte()):
                    #junk_height = pyro.sample("junk_height", dist.Gamma(junk_height_loc[j] * junk_height_beta[j], junk_height_beta[j]))
                    #junk_width = pyro.sample("junk_width", dist.Gamma(junk_width_loc[j] * junk_width_beta[j], junk_width_beta[j]))
                    junk_height = pyro.sample("junk_height", dist.Gamma(height_loc[j] * height_beta[j], height_beta[j]))
                    junk_width = pyro.sample("junk_width", dist.Gamma(width_loc[j] * width_beta[j], width_beta[j]))
                    junk_x0 = pyro.sample("junk_x0", self.Location(0., junk_x0_size[j], -(self.D+3)/2, self.D+3))
                    junk_y0 = pyro.sample("junk_y0", self.Location(0., junk_y0_size[j], -(self.D+3)/2, self.D+3))
            
                # return locs for K classes
                spots = self.gaussian_spot(batch_idx, height, width, x0, y0)
                junk = self.gaussian_spot(batch_idx, junk_height, junk_width, junk_x0, junk_y0)
                locs = torch.where(z > 0, spots, torch.zeros_like(spots)) + torch.where(j > 0, junk, torch.zeros_like(junk)) + background
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, **noise_params), obs=self.data[batch_idx])
    
    @config_enumerate
    def guide(self):
        # noise variables
        for var in self._params:
            guide_params = dict()
            for param in self._params[var]["guide_params"]:
                guide_params[param] = pyro.param(**self._params[var]["guide_params"][param]) 
            pyro.sample(var, self._params[var]["guide_dist"](**guide_params))

        # plates
        K_plate = pyro.plate("K_plate", self.K-1)
        N_plate = pyro.plate("N_plate", self.N, 16, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        junk_pi_concentration = pyro.param("junk_pi_concentration", torch.ones(2)*self.N*self.F/2, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.b_loc.mean(), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", 100*torch.ones(self.K-1), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K-1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", self.data.w_loc.mean()*torch.ones(self.K-1), constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1), constraint=constraints.positive)
        #x0_scale_v = pyro.param("x0_scale_v", torch.ones(self.K-1), constraint=constraints.positive)
        #y0_scale_v = pyro.param("y0_scale_v", torch.ones(self.K-1), constraint=constraints.positive)
        x0_size_v = pyro.param("x0_size_v", (((self.D+3)/(2*0.5))**2 - 1)*torch.ones(self.K-1), constraint=constraints.greater_than(2.))
        y0_size_v = pyro.param("y0_size_v", (((self.D+3)/(2*0.5))**2 - 1)*torch.ones(self.K-1), constraint=constraints.greater_than(2.))
        #x0_scale_v = pyro.param("x0_scale_v", torch.tensor([0.75]), constraint=constraints.positive)
        #y0_scale_v = pyro.param("y0_scale_v", torch.tensor([0.75]), constraint=constraints.positive)
        
        #junk_height_loc_v = pyro.param("junk_height_loc_v", 100*torch.ones(1), constraint=constraints.positive)
        #junk_height_beta_v = pyro.param("junk_height_beta_v", torch.ones(1)*10, constraint=constraints.positive)
        #junk_width_loc_v = pyro.param("junk_width_loc_v", self.data.w_loc.mean()*torch.ones(1), constraint=constraints.positive)
        #junk_width_beta_v = pyro.param("junk_width_beta_v", torch.ones(1), constraint=constraints.positive)

        # AoI & Frame Local Parameters
        b_loc = pyro.param("b_loc", self.data.b_loc.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", self.data.b_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        h_loc = pyro.param("h_loc", self.data.h_loc.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", self.data.h_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        w_loc = pyro.param("w_loc", self.data.w_loc.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        w_beta = pyro.param("w_beta", self.data.w_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        x_mode = pyro.param("x_mode", self.data.x_mode.reshape(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        x_size = pyro.param("x_size", self.data.x_size.reshape(self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        y_mode = pyro.param("y_mode", self.data.y_mode.reshape(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        y_size = pyro.param("y_size", self.data.y_size.reshape(self.N,self.F,1,1), constraint=constraints.greater_than(2.))

        jh_loc = pyro.param("jh_loc", self.data.h_loc.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        jh_beta = pyro.param("jh_beta", self.data.h_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        jw_loc = pyro.param("jw_loc", self.data.w_loc.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        jw_beta = pyro.param("jw_beta", self.data.w_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        jx_mode = pyro.param("jx_mode", self.data.x_mode.reshape(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        jx_size = pyro.param("jx_size", self.data.x_size.reshape(self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        jy_mode = pyro.param("jy_mode", self.data.y_mode.reshape(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        jy_size = pyro.param("jy_size", self.data.y_size.reshape(self.N,self.F,1,1), constraint=constraints.greater_than(2.))

        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        j_probs = pyro.param("j_probs", torch.ones(self.N,self.F,1,1,2) / 2, constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("junk_pi", dist.Dirichlet(junk_pi_concentration))
        pyro.sample("background_loc", dist.Delta(background_loc_v))
        pyro.sample("background_beta", dist.Delta(background_beta_v))
        with K_plate:
                pyro.sample("height_loc", dist.Delta(height_loc_v))
                pyro.sample("height_beta", dist.Delta(height_beta_v))
                pyro.sample("width_loc", dist.Delta(width_loc_v))
                pyro.sample("width_beta", dist.Delta(width_beta_v))
                pyro.sample("x0_size", dist.Delta(x0_size_v))
                pyro.sample("y0_size", dist.Delta(y0_size_v))
        #pyro.sample("junk_height_loc", dist.Delta(junk_height_loc_v))
        #pyro.sample("junk_height_beta", dist.Delta(junk_height_beta_v))
        #pyro.sample("junk_width_loc", dist.Delta(junk_width_loc_v))
        #pyro.sample("junk_width_beta", dist.Delta(junk_width_beta_v))
        
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(z_probs[batch_idx]))
                j = pyro.sample("j", dist.Categorical(j_probs[batch_idx]))
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta[batch_idx], b_beta[batch_idx]))
                with poutine.mask(mask=(z > 0).byte()):
                    pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta[batch_idx], h_beta[batch_idx]))
                    pyro.sample("width", dist.Gamma(w_loc[batch_idx] * w_beta[batch_idx], w_beta[batch_idx]))
                    pyro.sample("x0", self.Location(x_mode[batch_idx], x_size[batch_idx], -(self.D+3)/2, self.D+3))
                    pyro.sample("y0", self.Location(y_mode[batch_idx], y_size[batch_idx], -(self.D+3)/2, self.D+3))
                    #pyro.sample("x0", dist.Normal(x_loc[batch_idx], x_scale[batch_idx]))
                    #pyro.sample("y0", dist.Normal(y_loc[batch_idx], y_scale[batch_idx]))

                with poutine.mask(mask=j.byte()):
                    pyro.sample("junk_height", dist.Gamma(jh_loc[batch_idx] * jh_beta[batch_idx], jh_beta[batch_idx]))
                    pyro.sample("junk_width", dist.Gamma(jw_loc[batch_idx] * jw_beta[batch_idx], jw_beta[batch_idx]))
                    pyro.sample("junk_x0", self.Location(jx_mode[batch_idx], jx_size[batch_idx], -(self.D+3)/2, self.D+3))
                    pyro.sample("junk_y0", self.Location(jy_mode[batch_idx], jy_size[batch_idx], -(self.D+3)/2, self.D+3))
                    #pyro.sample("junk_x0", dist.Normal(jx_loc[batch_idx], jx_scale[batch_idx]))
                    #pyro.sample("junk_y0", dist.Normal(jy_loc[batch_idx], jy_scale[batch_idx]))
        
        #return height, width, background, x0, y0

    def fixed_epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.fixed.step()
            if not (self.epoch_count % 100):    
                write_summary(self.epoch_count, epoch_loss, self, self.fixed, self.writer, feature=False, mcc=False)
            if not (self.epoch_count % 1000):    
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()


    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.svi.step()
            if not (self.epoch_count % 100):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False, mcc=False)
            if not (self.epoch_count % 1000):    
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        self.optim.save(os.path.join(self.data.path, "runs", "classifier", 
                "{}".format(self.__name__), "K{}".format(self.K), "optimizer"))
        pyro.get_param_store().save(os.path.join(self.data.path, "runs", "classifier", 
                "{}".format(self.__name__), "K{}".format(self.K), "params"))
        np.savetxt(os.path.join(self.data.path, "runs", "classifier", 
                "{}".format(self.__name__), "K{}".format(self.K), "epoch_count"), np.array([self.epoch_count]))
        #for p in pyro.get_param_store().get_all_param_names():
        #    torch.save(pyro.param(p).detach().squeeze(), os.path.join(self.data.path, "runs", "features", "{}.pt".format(p)))
        if verbose:
            print("Classification results were saved in {}...".format(self.data.path))

    def load(self):
        self.epoch_count = int(np.loadtxt(os.path.join(self.data.path, "runs", "classifier", 
                "{}".format(self.__name__), "K{}".format(self.K), "epoch_count")))
        self.optim.load(os.path.join(self.data.path, "runs", "classifier", 
                "{}".format(self.__name__), "K{}".format(self.K), "optimizer"))
        pyro.get_param_store().load(os.path.join(self.data.path, "runs", "classifier", 
                "{}".format(self.__name__), "K{}".format(self.K), "params"))
