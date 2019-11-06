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
    if param_name in ["size", "w_size"]:
        return {"lr": 0.00025}
    else:
        return {"lr": 0.00005}

class JExtraction:
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, lr, jit, noise="GammaOffset"):
        self.data = data
        self.dataset = dataset
        self.N, self.F, self.D, _ = data._store.shape
        self.K = 2
        self.M = 2
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).to(torch.float32)
        #self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,1,2)
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(self.N,self.F,1,1,1,1,2).repeat(1,1,1,1,self.K,self.K,1)# N,F,1,1,M,K,2
        #self.spot_scale = torch.eye(2).reshape(1,1,1,1,1,1,2,2)
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)

        pyro.clear_param_store()
        self.epoch_count = 0
        #self.optim = pyro.optim.Adam(per_param_args)
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = JitTrace_ELBO() if jit else Trace_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["w_mode", "w_size"]), 
                                        self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "junk", "M{}".format(self.M)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0, k):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = self.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
        spot_locs[...,0] += x0 # N,F,D,D,M,K,2 .permute(1,2,3,4,5,0) # adjust for the center of the first frame
        spot_locs[...,1] += y0 #.permute(1,2,3,4,5,0) # x0 and y0 can be either scalars or vectors
        # N,F,D,D,M,K -> tril
        #height[...,1,:] = 0
        height = height[...,k,:k+1]
        if k == 0:
            #spot_locs = spot_locs.reshape(len(batch_idx),self.F,1,1,2) 
            spot_locs = spot_locs[...,0,0,:]
            width = width.reshape(1,1,1,1)
            rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D,M
            height = height[...,0]
            return height * gaussian_spot # N,F,D,D,M
        else:
            spot_locs = spot_locs[...,k,:k+1,:]
            #height = height[...,self.M-1,:self.M]
            p = height / height.sum(dim=-1, keepdims=True)
            logits = torch.log(p/(1-p))
            #logits = logits # N,F,D,D,M,K
            #w = width.unsqueeze(dim=-1).repeat(1,1,1,1,1,1,2) # N,F,D,D,M,K,2
            #w = width.reshape(1,1,1,1,1,1,1).repeat(len(batch_idx),self.F,1,1,2,2,2) # N,F,D,D,M,K,2
            w = width.reshape(1,1,1,1,1,1).repeat(len(batch_idx),self.F,1,1,k+1,2) # N,F,D,D,M,K,2
            rv = dist.MixtureOfDiagNormals(locs=spot_locs, coord_scale=w, component_logits=logits)
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D,M
            return height.sum(dim=-1) * gaussian_spot # N,F,D,D,M
    
    def Location(self, mode, size, loc, scale):
        mode = (mode - loc) / scale
        concentration1 = mode * (size - 2) + 1
        concentration0 = (1 - mode) * (size - 2) + 1
        base_distribution = dist.Beta(concentration1, concentration0)
        transforms =  [AffineTransform(loc=loc, scale=scale)]
        return dist.TransformedDistribution(base_distribution, transforms)

    def model(self):
        # noise variables
        #noise_params = dict()
        #for var in self._params:
        #    noise_params[var] = pyro.sample(var, self._params[var]["prior"])
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(torch.tensor(0.), offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(torch.tensor(50.)))

        N_plate = pyro.plate("N_plate", self.N, subsample_size=16, dim=-2)
        F_plate = pyro.plate("F_plate", size=self.F, dim=-1)
        
        x0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])
        y0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])
        #x0_size = torch.tensor(2.)
        #y0_size = torch.tensor(2.)

        width = pyro.sample("width", self.Location(1.3, 4., 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1]).to_event(2))
                tril_mask = torch.ones(1,1,1,1,self.K,self.K).tril()
                #tril_mask[...,self.M-1,:] = 0
                tril_mask = tril_mask.byte()
                height = pyro.sample("height", dist.HalfNormal(3000.).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4)) # N,F,1,1,M,K
                #width = pyro.sample("width", dist.Gamma(1., 0.1).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4))
                #width = pyro.sample("width", self.Location(1.3, 4., 0.5, 2.5).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4))
                x0 = pyro.sample("x0", self.Location(0., x0_size, -(self.D+3)/2, self.D+3).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4)) # N,F,1,1,M,K
                y0 = pyro.sample("y0", self.Location(0., y0_size, -(self.D+3)/2, self.D+3).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4)) # N,F,1,1,M,K

                for k in range(self.K):
                    spot = self.gaussian_spot(batch_idx, height, width, x0, y0, k)
                    locs = spot + background
                #locs = spot[...,0] + background
                    pyro.sample("data_{}".format(k), self.CameraUnit(locs, gain, offset).to_event(2), obs=self.data[batch_idx])
                #    locs = spot[...,k] + background
                #    pyro.sample("data_{}".format(k), self.CameraUnit(locs, gain, offset).to_event(2), obs=self.data[batch_idx])
                    #pyro.sample("data_{}".format(k), self.CameraUnit(locs, **noise_params).to_event(2), obs=self.data[batch_idx])

    def guide(self):
        # noise variables
        #for var in self._params:
            #guide_params = dict()
            #for param in self._params[var]["guide_params"]:
                #guide_params[param] = pyro.param(**self._params[var]["guide_params"][param]) 
            #pyro.sample(var, self._params[var]["guide_dist"](**guide_params))
        offset_max = self.data._store.min() - 0.1
        offset_v = pyro.param("offset_v", offset_max-50, constraint=constraints.interval(0,offset_max.item()))
        gain_v = pyro.param("gain_v", torch.tensor(5.), constraint=constraints.positive)
        pyro.sample("offset", dist.Delta(offset_v))
        pyro.sample("gain", dist.Delta(gain_v))

        N_plate = pyro.plate("N_plate", self.N, subsample_size=16, dim=-2)
        F_plate = pyro.plate("F_plate", size=self.F, dim=-1)
        
        # global locs variables
        b_loc = pyro.param("b_loc", self.data.background.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", torch.ones(1)*10., constraint=constraints.positive)
        
        # local variables
        #w_mode = pyro.param("w_mode", torch.ones(self.N,self.F,1,1,self.K,self.K)*1.5, constraint=constraints.interval(0.5,2.5))
        w_mode = pyro.param("w_mode", torch.ones(1)*1.35, constraint=constraints.interval(0.5,3.))
        w_size = pyro.param("w_size", torch.ones(1)*100., constraint=constraints.greater_than(2.))
        #w_loc = pyro.param("w_loc", torch.ones(1)*1.5, constraint=constraints.positive)
        #w_beta = pyro.param("w_beta", torch.ones(1)*100., constraint=constraints.positive)
        h_loc = pyro.param("h_loc", 1000*torch.ones(self.N,self.F,1,1,self.K,self.K), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.N,self.F,1,1,self.K,self.K), constraint=constraints.positive)
        #r_mode = pyro.param("r_mode", torch.zeros(self.N,self.F,1,1,self.K,self.K), constraint=constraints.interval(0,(self.D**2-2*self.F+1)/4))
        #phi_mode = pyro.param("phi_mode", torch.zeros(self.N,self.F,1,1,self.K,self.K), constraint=constraints.interval(0,2*math.pi))
        #x_mode = r_mode * torch.sin(phi_mode)
        #y_mode = r_mode * torch.cos(phi_mode)
        x_mode = pyro.param("x_mode", torch.zeros(self.N,self.F,1,1,self.K,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #mode = torch.zeros(self.N,self.F,1,1,self.K,self.K)
        #mode[...,1,0] += 2.5
        #mode[...,1,1] -= 2.5
        y_mode = pyro.param("y_mode", torch.zeros(self.N,self.F,1,1,self.K,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        intensity = torch.ones(self.N,self.F,1,1,self.K,self.K)*10.
        intensity[...,1] = (((self.D+3)/(2*0.5))**2 - 1)
        size = pyro.param("size", intensity, constraint=constraints.greater_than(2.))
        #x_size = pyro.param("x_size", psize, constraint=constraints.greater_than(2.))
        #y_size = pyro.param("y_size", psize, constraint=constraints.greater_than(2.))
        
        pyro.sample("width", self.Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta).to_event(2))
                tril_mask = torch.ones(len(batch_idx),self.F,1,1,self.K,self.K).tril()
                #tril_mask[...,self.M-1,:] = 0
                tril_mask = tril_mask.byte()
                pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta[batch_idx], h_beta[batch_idx]).mask(tril_mask).to_event(4))
                #pyro.sample("width", dist.Gamma(w_loc[batch_idx] * w_beta * size[batch_idx], w_beta * size[batch_idx]).mask(tril_mask).to_event(4))
                #pyro.sample("width", self.Location(w_mode, w_size[batch_idx], 0.5, 2.5).mask(tril_mask).to_event(4))
                #pyro.sample("width", dist.Gamma(w_loc[batch_idx] * w_beta * size[batch_idx], w_beta * size[batch_idx]).mask(tril_mask).to_event(4))
                pyro.sample("x0", self.Location(x_mode[batch_idx], size[batch_idx], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4))
                pyro.sample("y0", self.Location(y_mode[batch_idx], size[batch_idx], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4))
                #pyro.sample("x0", self.Location(x_mode[batch_idx], size[batch_idx], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4))
                #pyro.sample("y0", self.Location(y_mode[batch_idx], size[batch_idx], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4))

    def fixed_epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.fixed.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.fixed, self.writer, feature=False, mcc=False)
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.svi.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=True, mcc=False)
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        self.optim.save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "M{}".format(self.M), "optimizer"))
        pyro.get_param_store().save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "M{}".format(self.M), "params"))
        np.savetxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "M{}".format(self.M), "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("Features were extracted and saved in {}.".format(self.data.path))

    def load(self):
        self.epoch_count = int(np.loadtxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "M{}".format(self.M), "epoch_count")))
        self.optim.load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "M{}".format(self.M), "optimizer"))
        pyro.get_param_store().load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "junk", "M{}".format(self.M), "params"))
