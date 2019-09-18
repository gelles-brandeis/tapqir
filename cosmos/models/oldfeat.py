import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn

class FeatureExtraction:
    """ Gaussian Spot Model """
    def __init__(self, data, lr, noise="GammaOffset"):
        self.data = data
        self.N, self.F, self.D, _ = data._store.shape
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).to(torch.float32)
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(self.N,self.F,1,1,2)
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)

        pyro.clear_param_store()
        self.epoch_count = 0
        self.optim = pyro.optim.AdamW({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = JitTrace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "features"))
        
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
        spot_locs = self.target_locs[batch_idx] # select target locs for given indices
        spot_locs[...,0] += x0 # adjust for the center of the first frame
        spot_locs[...,1] += y0 # x0 and y0 can be either scalars or vectors
        rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        return height * gaussian_spot #
    
    #def model(self, batch_idx):
    def model(self):
        # noise variables
        noise_params = dict()
        for var in self._params:
            noise_params[var] = pyro.sample(var, self._params[var]["prior"])

        #N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        N_plate = pyro.plate("N_plate", self.N, 16, dim=-4)
        F_plate = pyro.plate("F_plate", size=self.F, dim=-3)
        
        with N_plate as batch_idx:
            with F_plate:
                background = pyro.sample("background", dist.HalfNormal(1000.))
                height = pyro.sample("height", dist.HalfNormal(500.))
                width = pyro.sample("width", dist.Gamma(1, 0.1))
                x0 = pyro.sample("x0", self.Location(0., 3., -(self.D+3)/2, self.D+3))
                y0 = pyro.sample("y0", self.Location(0., 3., -(self.D+3)/2, self.D+3))
                #x0 = pyro.sample("x0", dist.Normal(0.,10.))
                #y0 = pyro.sample("y0", dist.Normal(0.,10.))

                locs = self.gaussian_spot(batch_idx, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, **noise_params), obs=self.data[batch_idx])


    def guide(self):
        # noise variables
        for var in self._params:
            guide_params = dict()
            for param in self._params[var]["guide_params"]:
                guide_params[param] = pyro.param(**self._params[var]["guide_params"][param]) 
            pyro.sample(var, self._params[var]["guide_dist"](**guide_params))

        N_plate = pyro.plate("N_plate", self.N, 16, dim=-4)
        F_plate = pyro.plate("F_plate", size=self.F, dim=-3)
        
        # global locs variables
        b_loc = pyro.param("b_loc", self.data.background.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", torch.ones(self.N,self.F,1,1), constraint=constraints.positive)
        
        # local variables
        w_loc = pyro.param("w_loc", torch.ones(self.N,self.F,1,1)*1.5, constraint=constraints.positive)
        w_beta = pyro.param("w_beta", torch.ones(self.N,self.F,1,1), constraint=constraints.positive)
        #x_loc = pyro.param("x_loc", torch.zeros(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #x_scale = pyro.param("x_scale", torch.ones(self.N,self.F,1,1), constraint=constraints.interval(0, (self.D+3)/2))
        #y_loc = pyro.param("y_loc", torch.zeros(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #y_scale = pyro.param("y_scale", torch.ones(self.N,self.F,1,1), constraint=constraints.interval(0, (self.D+3)/2))
        h_loc = pyro.param("h_loc", self.data.height.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.N,self.F,1,1), constraint=constraints.positive)
        x_mode = pyro.param("x_mode", torch.zeros(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        x_size = pyro.param("x_size", 1000 * torch.ones(self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        y_mode = pyro.param("y_mode", torch.zeros(self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        y_size = pyro.param("y_size", 1000 * torch.ones(self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        
        with N_plate as batch_idx:
            with F_plate:
                # local height and locs
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta[batch_idx], b_beta[batch_idx]))
                pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta[batch_idx], h_beta[batch_idx]))
                pyro.sample("width", dist.Gamma(w_loc[batch_idx] * w_beta[batch_idx], w_beta[batch_idx]))
                pyro.sample("x0", self.Location(x_mode[batch_idx], x_size[batch_idx], -(self.D+3)/2, self.D+3))
                pyro.sample("y0", self.Location(y_mode[batch_idx], y_size[batch_idx], -(self.D+3)/2, self.D+3))
                #pyro.sample("x0", dist.Normal(x_loc[batch_idx], x_scale[batch_idx]))
                #pyro.sample("y0", dist.Normal(y_loc[batch_idx], y_scale[batch_idx]))
        
    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.svi.step()
            if not (self.epoch_count % 250):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=True, mcc=False)
            if not (self.epoch_count % 1000):    
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        self.optim.save(os.path.join(self.data.path, "runs", "features", "optimizer"))
        pyro.get_param_store().save(os.path.join(self.data.path, "runs", "features", "params"))
        np.savetxt(os.path.join(self.data.path, "runs", "features", "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("Features were extracted and saved in {}.".format(self.data.path))

    def load(self):
        self.epoch_count = int(np.loadtxt(os.path.join(self.data.path, "runs", "features", "epoch_count")))
        self.optim.load(os.path.join(self.data.path, "runs", "features", "optimizer"))
        pyro.get_param_store().load(os.path.join(self.data.path, "runs", "features", "params"))
