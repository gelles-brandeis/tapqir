import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn

class Model:
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
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = Trace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "features"))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = self.target_locs[batch_idx] # select target locs for given indices
        spot_locs[...,0] += x0 # adjust for the center of the first frame
        spot_locs[...,1] += y0 # x0 and y0 can be either scalars or vectors
        rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        return height * gaussian_spot #
    
    def model(self, batch_idx):
        raise NotImplementedError


    def guide(self, batch_idx):
        raise NotImplementedError
        
    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                loss = self.svi.step(batch_idx)
                epoch_loss += loss * len(batch_idx) / self.N
            if not (self.epoch_count % 10):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=True, mcc=False)
            if not (self.epoch_count % 100):    
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        for p in pyro.get_param_store().get_all_param_names():
            torch.save(pyro.param(p).detach().squeeze(), os.path.join(self.data.path, "runs", "features", "{}.pt".format(p)))
        if verbose:
            print("Features were extracted and saved in {}.".format(self.data.path))
