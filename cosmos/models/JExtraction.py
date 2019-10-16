import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
import pyro.distributions as dist
#from pyro.infer import SVI, Trace_ELBO
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn

def per_param_args(module_name, param_name):
    if param_name in ["size", "jsize"]:
        return {"lr": 0.002}
    else:
        return {"lr": 0.0005}

class JExtraction:
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, lr, jit, noise="GammaOffset"):
        self.data = data
        self.dataset = dataset
        self.N, self.F, self.D, _ = data._store.shape
        self.K = 2
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).to(torch.float32)
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(1,self.N,self.F,1,1,2).repeat(self.K,1,1,1,1,1).permute(1,2,3,4,0,5) # N,F,1,1,K,2
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)

        pyro.clear_param_store()
        self.epoch_count = 0
        #self.optim = pyro.optim.Adam(per_param_args)
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = JitTrace_ELBO() if jit else Trace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "junk", "K{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = self.target_locs[batch_idx] # select target locs for given indices
        spot_locs[...,0] += x0.permute(1,2,3,4,0) # adjust for the center of the first frame
        spot_locs[...,1] += y0.permute(1,2,3,4,0) # x0 and y0 can be either scalars or vectors
        if self.K == 1:
            rv = dist.MultivariateNormal(spot_locs.reshape(len(batch_idx),self.F,1,1,2), scale_tril=self.spot_scale * width.reshape(len(batch_idx),self.F,1,1).view(width.size()+(1,1)))
        if self.K > 1:
            p = height / height.sum(dim=0)
            logits = torch.log(p/(1-p))
            logits = logits.permute(1,2,3,4,0)
            w = width.reshape(self.K,1,len(batch_idx),self.F,1,1).repeat(1,2,1,1,1,1).permute(2,3,4,5,0,1)
            rv = dist.MixtureOfDiagNormals(locs=spot_locs, coord_scale=w, component_logits=logits)
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) #* 2 * math.pi #* width**2
        return height.sum(dim=0) * gaussian_spot #
    
    def Location(self, mode, size, loc, scale):
        mode = (mode - loc) / scale
        concentration1 = mode * (size - 2) + 1
        concentration0 = (1 - mode) * (size - 2) + 1
        base_distribution = dist.Beta(concentration1, concentration0)
        transforms =  [AffineTransform(loc=loc, scale=scale)]
        return dist.TransformedDistribution(base_distribution, transforms)

    def model(self):
        # noise variables
        noise_params = dict()
        for var in self._params:
            noise_params[var] = pyro.sample(var, self._params[var]["prior"])

        K_plate = pyro.plate("K_plate", self.K, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample_size=16, dim=-4)
        F_plate = pyro.plate("F_plate", size=self.F, dim=-3)
        
        with N_plate as batch_idx:
            with F_plate:
                background = pyro.sample("background", dist.HalfNormal(1000.))
                with K_plate:
                    height = pyro.sample("height", dist.HalfNormal(500.)) # K,N,F,1,1
                    width = pyro.sample("width", dist.Gamma(1, 0.1))
                    x0 = pyro.sample("x0", self.Location(0., 2., -(self.D+3)/2, self.D+3)) # K,N,F,1,1
                    y0 = pyro.sample("y0", self.Location(0., 2., -(self.D+3)/2, self.D+3)) # K,N,F,1,1

                    spots = self.gaussian_spot(batch_idx, height, width, x0, y0)
                    locs = spots + background
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

        K_plate = pyro.plate("K_plate", self.K, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample_size=16, dim=-4)
        F_plate = pyro.plate("F_plate", size=self.F, dim=-3)
        
        # global locs variables
        b_loc = pyro.param("b_loc_{}".format(self.K), self.data.background.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta_{}".format(self.K), torch.ones(1), constraint=constraints.positive)
        
        # local variables
        w_loc = pyro.param("w_loc_{}".format(self.K), torch.ones(1)*1.5, constraint=constraints.positive)
        w_beta = pyro.param("w_beta_{}".format(self.K), torch.ones(1)*100, constraint=constraints.positive)
        #w_loc = pyro.param("w_loc", 1.5*torch.ones(self.K,self.N,self.F,1,1), constraint=constraints.positive)
        #w_beta = pyro.param("w_beta", 100*torch.ones(self.K,self.N,self.F,1,1), constraint=constraints.positive)
        h_loc = pyro.param("h_loc_{}".format(self.K), 100*torch.ones(self.K,self.N,self.F,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta_{}".format(self.K), torch.ones(self.K,self.N,self.F,1,1), constraint=constraints.positive)
        x_mode = pyro.param("x_mode_{}".format(self.K), torch.zeros(self.K,self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #x_size = pyro.param("x_size", 1000 * torch.ones(self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        if self.K == 1:
            size = pyro.param("size_{}".format(self.K), torch.tensor([10.]).reshape(self.K,1,1,1,1) * torch.ones(self.K,self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        elif self.K == 2:
            size = pyro.param("size_{}".format(self.K), torch.tensor([100., 10.]).reshape(self.K,1,1,1,1) * torch.ones(self.K,self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        y_mode = pyro.param("y_mode_{}".format(self.K), torch.zeros(self.K,self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #y_size = pyro.param("y_size", 1000 * torch.ones(self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        
        with N_plate as batch_idx:
            with F_plate:
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta))
                with K_plate:
                    # local height and locs
                    pyro.sample("height", dist.Gamma(h_loc[:,batch_idx] * h_beta[:,batch_idx], h_beta[:,batch_idx]))
                    pyro.sample("width", dist.Gamma(w_loc * w_beta * size[:,batch_idx], w_beta * size[:,batch_idx]))
                    pyro.sample("x0", self.Location(x_mode[:,batch_idx], size[:,batch_idx], -(self.D+3)/2, self.D+3))
                    pyro.sample("y0", self.Location(y_mode[:,batch_idx], size[:,batch_idx], -(self.D+3)/2, self.D+3))

    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.svi.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=True, mcc=False)
            if not (self.epoch_count % 1000):    
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        self.optim.save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "K{}".format(self.K), "optimizer"))
        pyro.get_param_store().save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "K{}".format(self.K), "params"))
        np.savetxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "K{}".format(self.K), "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("Features were extracted and saved in {}.".format(self.data.path))

    def load(self):
        self.epoch_count = int(np.loadtxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "K{}".format(self.K), "epoch_count")))
        self.optim.load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "K{}".format(self.K), "optimizer"))
        pyro.get_param_store().load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "K{}".format(self.K), "params"))
