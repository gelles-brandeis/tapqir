import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, Trace_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine
import pyro.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn

class Modelv2:
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
        #self.pixel_pos = torch.tensor(np.indices((self.D,self.D))).float()
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).to(torch.float32)
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
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["h_loc", "h_beta", "b_loc", "b_beta",
            "w_loc", "w_beta", "x_loc", "x_scale", "y_loc", "y_scale", "x0_scale_v", "y0_scale_v", "height_loc_v"]), self.optim, loss=self.elbo)
        self.prefit = SVI(self.model, poutine.block(self.guide, hide=["h_loc", "h_beta", "b_loc", "b_beta",
            "w_loc", "w_beta", "x_loc", "x_scale", "y_loc", "y_scale", "x0_scale_v", "y0_scale_v"]), self.optim, loss=self.elbo)
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        #self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "classifier", "K{}".format(self.K), "lr{}".format(self.lr)))
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "detector", "{}".format(self.__name__), "M{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    #@profile
    def gaussian_spot(self, batch_idx, frame_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        # select target locs for given indices
        spot_locs = self.target_locs[batch_idx][:,frame_idx] # ind,F,D,D,2
        # adjust for the center of the first frame
        # x0 and y0 can be either scalars or vectors
        spot_locs[...,0] += x0
        spot_locs[...,1] += y0
        rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        # height can be either a scalar or a vector
        return height * gaussian_spot #

    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        #plates
        K_plate = pyro.plate("K_plate", self.K)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1,1]))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(10.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(10.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(10.))

        
        with N_plate:
            with F_plate:
        #for n in pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4):
            # AoI Local Variables
            #for f in pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3):
                # AoI & Frame Local Variables
                #background = pyro.sample("background_{}_{}".format(n,f), dist.Gamma(background_loc*background_beta, background_beta))
                #z = pyro.sample("z_{}_{}".format(n,f), dist.Categorical(pi))
                #height = pyro.sample("height_{}_{}".format(n,f), dist.Gamma(height_loc[z]*height_beta[z], height_beta[z]))
                #width = pyro.sample("width_{}_{}".format(n,f), dist.Gamma(width_loc[z]*width_beta[z], width_beta[z]))
                #x0 = pyro.sample("x0_{}_{}".format(n,f), dist.Normal(0., x0_scale[z]))
                #y0 = pyro.sample("y0_{}_{}".format(n,f), dist.Normal(0., y0_scale[z]))
                background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta))
                z = pyro.sample("z", dist.Categorical(pi))
                height = pyro.sample("height", dist.Gamma(height_loc[z]*height_beta[z], height_beta[z]))
                width = pyro.sample("width", dist.Gamma(width_loc[z]*width_beta[z], width_beta[z]))
                x0 = pyro.sample("x0", dist.Normal(0., x0_scale[z]))
                #x0 = pyro.sample("x0", dist.Normal(0., x0_scale[z]).mask(z == 2))
                y0 = pyro.sample("y0", dist.Normal(0., y0_scale[z]))

        # return locs for K classes
        locs = self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0) + background

        return locs, N_plate, F_plate
    
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        # plates
        K_plate = pyro.plate("K_plate", self.K)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        
        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", 
                torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", 
                self.data.b.mean(), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", 
                torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", 
                #2*self.data.h.mean()*torch.rand(self.K), constraint=constraints.positive)
                torch.tensor([20., 120., 100.]), constraint=constraints.positive)
        #height_loc_v = pyro.param("height_loc_v", 
        #        torch.tensor([10., 100., 100.]), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", 
                10*torch.ones(self.K), constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", 
                self.data.w.mean()*torch.ones(self.K), constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", 
                torch.ones(self.K), constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", 
                #torch.rand(self.K), constraint=constraints.positive)
                torch.tensor([15., 0.5, 15.]), constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", 
                #torch.rand(self.K), constraint=constraints.positive)
                torch.tensor([15., 0.5, 15.]), constraint=constraints.positive)
        
        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.b.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", self.data.b_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        h_loc = pyro.param("h_loc", self.data.h.reshape(self.N,self.F,1,1), constraint=constraints.positive) 
        h_beta = pyro.param("h_beta", self.data.h_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive) 
        w_loc = pyro.param("w_loc", self.data.w.reshape(self.N,self.F,1,1)*1.5, constraint=constraints.positive)
        w_beta = pyro.param("w_beta", self.data.w_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        x_loc = pyro.param("x_loc", self.data.x0.reshape(self.N,self.F,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", self.data.x0_scale.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        y_loc = pyro.param("y_loc", self.data.y0.reshape(self.N,self.F,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", self.data.y0_scale.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        
        # AoI & Frame Local Parameters
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        #z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("background_loc", dist.Delta(background_loc_v))
        pyro.sample("background_beta", dist.Delta(background_beta_v))
        with K_plate:
            pyro.sample("height_loc", dist.Delta(height_loc_v))
            pyro.sample("height_beta", dist.Delta(height_beta_v))
            pyro.sample("width_loc", dist.Delta(width_loc_v))
            pyro.sample("width_beta", dist.Delta(width_beta_v))
            pyro.sample("x0_scale", dist.Delta(x0_scale_v))
            pyro.sample("y0_scale", dist.Delta(y0_scale_v))
            #with N_plate:
                                
        
        with N_plate:
            with F_plate:
        #for n in pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4):
            # AoI Local Variables
            #for f in pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3):
                # AoI & Frame Local Variables
                #z = pyro.sample("z_{}_{}".format(n,f), dist.Categorical(z_probs[n,f]))
                #pyro.sample("background_{}_{}".format(n,f), dist.Gamma(b_loc[n,f]*b_beta[n,f], b_beta[n,f]))
                #pyro.sample("height_{}_{}".format(n,f), dist.Gamma(h_loc[n,f]*h_beta[n,f], h_beta[n,f]))
                #pyro.sample("width_{}_{}".format(n,f), dist.Gamma(w_loc[n,f]*w_beta[n,f], w_beta[n,f]))
                #pyro.sample("x0_{}_{}".format(n,f), dist.Normal(x_loc[n,f], x_scale[n,f]))
                #pyro.sample("y0_{}_{}".format(n,f), dist.Normal(y_loc[n,f], y_scale[n,f]))
                z = pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                pyro.sample("background", dist.Gamma(b_loc[batch_idx][:,frame_idx]*b_beta[batch_idx][:,frame_idx], b_beta[batch_idx][:,frame_idx]))
                pyro.sample("height", dist.Gamma(h_loc[batch_idx][:,frame_idx]*h_beta[batch_idx][:,frame_idx], h_beta[batch_idx][:,frame_idx]))
                pyro.sample("width", dist.Gamma(w_loc[batch_idx][:,frame_idx]*w_beta[batch_idx][:,frame_idx], w_beta[batch_idx][:,frame_idx]))
                pyro.sample("x0", dist.Normal(x_loc[batch_idx][:,frame_idx], x_scale[batch_idx][:,frame_idx]))
                pyro.sample("y0", dist.Normal(y_loc[batch_idx][:,frame_idx], y_scale[batch_idx][:,frame_idx]))

    def fixed_epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                for frame_idx in DataLoader(Sampler(self.F), batch_size=self.F, shuffle=True):
                    loss = self.fixed.step(batch_idx, frame_idx)
                    epoch_loss += loss * len(batch_idx) * len(frame_idx) / (self.N * self.F)
            if not (self.epoch_count % 5):    
                write_summary(self.epoch_count, epoch_loss, self, self.fixed, self.writer, feature=False, mcc=False)
            self.epoch_count += 1

    def prefit_epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                for frame_idx in DataLoader(Sampler(self.F), batch_size=self.F, shuffle=True):
                    loss = self.prefit.step(batch_idx, frame_idx)
                    epoch_loss += loss * len(batch_idx) * len(frame_idx) / (self.N * self.F)
            if not (self.epoch_count % 5):    
                write_summary(self.epoch_count, epoch_loss, self, self.prefit, self.writer, feature=False, mcc=False)
            self.epoch_count += 1

    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                for frame_idx in DataLoader(Sampler(self.F), batch_size=self.F, shuffle=True):
                    loss = self.svi.step(batch_idx, frame_idx)
                    epoch_loss += loss * len(batch_idx) * len(frame_idx) / (self.N * self.F)
            if not (self.epoch_count % 5):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False, mcc=False)
            self.epoch_count += 1

    def save(self):
        torch.save(pyro.param("z_probs").detach().squeeze(), os.path.join(self.data.path, "runs", "classifier", 
            #"K{}".format(self.K), "lr{}".format(self.lr), "probs.pt"))
            "K{}".format(self.K), "prefit", "probs.pt"))
        print("Classification results were saved in {}...".format(self.data.path))
