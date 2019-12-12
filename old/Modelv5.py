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
from pyro.optim import Adam
import pyro.poutine as poutine
import pyro.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cosmos.utils.utils import write_summary
from cosmos.utils.glimpse_reader import Sampler
from cosmos.models.noise import _noise, _noise_fn

def per_param_args(module_name, param_name):
    if param_name in ["size", "jsize"]:
        return {"lr": 0.005}
    else:
        return {"lr": 0.002}

class Modelv5:
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, n_batch, jit, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "v5"
        self.data = data
        self.dataset = dataset
        self.N, self.F, self.D, _ = data._store.shape
        assert K >= 2
        self.K = K
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        self.h_loc = 1500 
        self.lr = lr
        self.n_batch = n_batch
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).to(torch.float32)
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(self.N,self.F,1,1,1,2).repeat(1,1,1,1,self.K,1)# N,F,1,1,M,K,2
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)
        
        pyro.clear_param_store()
        #pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "junk", "M2/lr/0.0005/h/10", "params"))
        self.epoch_count = 0
        #self.prefit_optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        #self.optim = pyro.optim.Adam(per_param_args)
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["h_loc", "h_beta", "w_mode", "w_size", "x_mode", "y_mode", "size", "b_loc", "b_beta", "gain_v", "offset_v"]), 
                                self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "detector", "{}".format(self.__name__), "M{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0, k):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = self.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
        spot_locs[...,0] += x0 # N,F,D,D,M,K,2 .permute(1,2,3,4,5,0) # adjust for the center of the first frame
        spot_locs[...,1] += y0 #.permute(1,2,3,4,5,0) # x0 and y0 can be either scalars or vectors
        # N,F,D,D,M,K -> tril
        #height[...,1,:] = 0
        #spot = torch.zeros(len(batch_idx),self.F,self.D,self.D,self.K+1)
        height = height[...,:k+1]
        if k == 0:
            #spot_locs = spot_locs.reshape(len(batch_idx),self.F,1,1,2) 
            spot_locs = spot_locs[...,0,:]
            #width = width[...,0]
            width = width.reshape(1,1,1,1)
            rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D,M
            height = height[...,0]
            return height * gaussian_spot # N,F,D,D,M
        else:
            spot_locs = spot_locs[...,:k+1,:]
            #height = height[...,self.M-1,:self.M]
            p = height / height.sum(dim=-1, keepdims=True)
            logits = torch.log(p/(1-p))
            #logits = logits # N,F,D,D,M,K
            #w = width.unsqueeze(dim=-1).repeat(1,1,1,1,1,1,2) # N,F,D,D,M,K,2
            w = width.reshape(1,1,1,1,1,1).repeat(len(batch_idx),self.F,1,1,k+1,2) # N,F,D,D,M,K,2
            #w = width[...,:k+1].reshape(len(batch_idx),self.F,1,1,k+1,1).repeat(1,1,1,1,1,2) # N,F,D,D,M,K,2
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
    
    @config_enumerate
    def model(self):
        # noise variables
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(torch.tensor(0.), offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(torch.tensor(50.)))

        #plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-2)
        F_plate = pyro.plate("F_plate", self.F, dim=-1)
        
        # Global Variables
        m_pi = pyro.sample("m_pi", dist.Dirichlet(0.5 * torch.ones(self.K+1)))
        #background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.))
        #background_beta = pyro.sample("background_beta", dist.HalfNormal(100.))
        
        #with K_plate:
        #height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
        #height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
        #width_mode = pyro.sample("width_mode", self.Location(1.3, 4., 0.5, 2.5))
        #width_size = pyro.sample("width_size", dist.HalfNormal(500.))
                
        x0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])
        y0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])
        #x0_size = 2. * torch.ones(1)
        #y0_size = 2. * torch.ones(1)


        width = pyro.sample("width", self.Location(1.3, 4., 0.5, 2.5))
        with N_plate as batch_idx:
            nind, find, xind, yind = torch.meshgrid(torch.arange(len(batch_idx)),torch.arange(self.F),torch.arange(self.D),torch.arange(self.D))
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1]).to_event(2))
                #background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta).expand([1,1,1,1]).to_event(2))
                m = pyro.sample("m", dist.Categorical(m_pi)) # N,F,1,1
                spot = torch.zeros(len(batch_idx),self.F,self.D,self.D,self.K+1)
                for k in range(self.K):
                    with poutine.mask(mask=(m == k+1).byte()):
                        tril_mask = torch.ones(1,1,1,1,self.K,self.K).tril().byte()[...,k,:]
                        height = pyro.sample("height_{}".format(k), dist.Gamma(pyro.param("height_loc") * pyro.param("height_beta"), pyro.param("height_beta")).expand([1,1,1,1,self.K]).mask(tril_mask).to_event(3)) # N,F,1,1,M,K
                        #width = pyro.sample("width_{}".format(k), self.Location(1.4, 4., 0.5, 2.5).expand([1,1,1,1,self.K]).mask(tril_mask).to_event(3))
                        #height = pyro.sample("height", dist.Gamma(height_loc * height_beta, height_beta).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4))
                        #width = pyro.sample("width", self.Location(width_mode, width_size, 0.5, 2.5).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4))
                        x0 = pyro.sample("x0_{}".format(k), self.Location(0., x0_size, -(self.D+3)/2, self.D+3).expand([1,1,1,1,self.K]).mask(tril_mask).to_event(3))
                        y0 = pyro.sample("y0_{}".format(k), self.Location(0., y0_size, -(self.D+3)/2, self.D+3).expand([1,1,1,1,self.K]).mask(tril_mask).to_event(3))

                        spot[...,k+1] = self.gaussian_spot(batch_idx, height, width, x0, y0, k)
                locs = spot[nind,find,xind,yind,m.view(m.size()+(1,1))] + background
                pyro.sample("data", self.CameraUnit(locs, gain, offset).to_event(2), obs=self.data[batch_idx])
    
    @config_enumerate
    def guide(self):
        offset_max = self.data._store.min() - 0.1
        offset_v = pyro.param("offset_v", offset_max-50, constraint=constraints.interval(0,offset_max.item()))
        gain_v = pyro.param("gain_v", torch.tensor(5.), constraint=constraints.positive)
        pyro.sample("offset", dist.Delta(offset_v))
        pyro.sample("gain", dist.Delta(gain_v))

        ######
        #h_max = np.percentile(pyro.param("h_loc").detach().cpu()[...,0,0], 95)
        #p1 = torch.where(pyro.param("h_loc").detach()[...,0,0] < h_max, pyro.param("h_loc").detach()[...,0,0]/h_max, torch.tensor(1.))

        #m2 = p1 * 0.35 + 0.1
        #m1 = p1 * 0.35 + 0.1
        #m0 = 1 - m1 - m2 

        #m_probs = torch.stack((m0,m1,m2), dim=-1)

        # plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-2)
        F_plate = pyro.plate("F_plate", self.F, dim=-1)

        # Global Parameters
        m_pi_concentration = pyro.param("m_pi_concentration", torch.ones(self.K+1)*self.N*self.F/(self.K+1), constraint=constraints.positive)
        #background_loc_loc = pyro.param("background_loc_loc", self.data.b_loc.mean()*torch.ones(1), constraint=constraints.positive)
        #background_loc_loc = pyro.param("background_loc_loc", self.data.background.mean(), constraint=constraints.positive)
        #background_loc_beta = pyro.param("background_loc_beta", torch.ones(1)*500, constraint=constraints.positive)
        #background_beta_loc = pyro.param("background_beta_loc", pyro.param("b_loc").detach().mean() / pyro.param("b_loc").detach().var(), constraint=constraints.positive)
        #background_beta_loc = pyro.param("background_beta_loc", torch.ones(1), constraint=constraints.positive)
        #background_beta_beta = pyro.param("background_beta_beta", torch.ones(1)*500, constraint=constraints.positive)
        #height_loc_loc = pyro.param("height_loc_loc", 1000*torch.ones(1), constraint=constraints.positive)
        #height_loc_beta = pyro.param("height_loc_beta", torch.ones(1)*100, constraint=constraints.positive)
        #height_beta_loc = pyro.param("height_beta_loc", torch.ones(1), constraint=constraints.positive)
        #height_beta_beta = pyro.param("height_beta_beta", torch.ones(1)*500, constraint=constraints.positive)
        #width_mode_v = pyro.param("width_mode_v", 1.3*torch.ones(1), constraint=constraints.interval(0.5,3.))
        #width_size_v = pyro.param("width_size_v", torch.ones(1)*100, constraint=constraints.greater_than(2.))

        #b_loc, b_beta= pyro.param("b_loc"), pyro.param("b_beta")
        #w_mode, w_size= pyro.param("w_mode"), pyro.param("w_size")
        #h_loc, h_beta= pyro.param("h_loc"), pyro.param("h_beta")
        #x_mode, y_mode, size= pyro.param("x_mode"), pyro.param("y_mode"), pyro.param("size")

        pyro.param("height_loc", 1500.*torch.ones(1), constraint=constraints.positive)
        pyro.param("height_beta", 0.005*torch.ones(1), constraint=constraints.positive)

        b_loc = pyro.param("b_loc", self.data.background.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", torch.ones(1)*10., constraint=constraints.positive)
        #w_mode = pyro.param("w_mode", torch.ones(self.N,self.F,1,1,self.K,self.K)*1.35, constraint=constraints.interval(0.5,3.))
        #w_size = pyro.param("w_size", torch.ones(self.N,self.F,1,1,self.K,self.K)*100., constraint=constraints.greater_than(2.))
        w_mode = pyro.param("w_mode", torch.ones(1)*1.35, constraint=constraints.interval(0.5,3.))
        w_size = pyro.param("w_size", torch.ones(1)*100., constraint=constraints.greater_than(2.))
        h_loc = pyro.param("h_loc", self.h_loc*torch.ones(self.N,self.F,1,1,self.K,self.K), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.N,self.F,1,1,self.K,self.K), constraint=constraints.positive)
        x_mode = pyro.param("x_mode", torch.zeros(self.N,self.F,1,1,self.K,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        y_mode = pyro.param("y_mode", torch.zeros(self.N,self.F,1,1,self.K,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        intensity = torch.ones(self.N,self.F,1,1,self.K,self.K)*(((self.D+3)/(2*0.5))**2 - 1)
        #intensity[...,1] = (((self.D+3)/(2*0.5))**2 - 1)
        size = pyro.param("size", intensity, constraint=constraints.greater_than(2.))

        m_probs = torch.zeros(self.N,self.F,self.K+1)
        m_probs[...,0] = 0.8
        m_probs[...,1] = 0.1
        m_probs[...,2] = 0.1
        #m_probs = pyro.param("m_probs", torch.ones(self.N,self.F,self.K+1) / (self.K+1), constraint=constraints.simplex)
        m_probs = pyro.param("m_probs", m_probs.reshape(self.N,self.F,self.K+1), constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("m_pi", dist.Dirichlet(m_pi_concentration))
        #pyro.sample("junk_pi", dist.Dirichlet(junk_pi_concentration))
        #pyro.sample("background_loc", dist.Gamma(background_loc_loc * background_loc_beta, background_loc_beta))
        #pyro.sample("background_beta", dist.Gamma(background_beta_loc * background_beta_beta, background_beta_beta))
        #with K_plate:
        #pyro.sample("height_loc", dist.Gamma(height_loc_loc * height_loc_beta, height_loc_beta))
        #pyro.sample("height_beta", dist.Gamma(height_beta_loc * height_beta_beta, height_beta_beta))
        #pyro.sample("width_mode", dist.Delta(width_mode_v))
        #pyro.sample("width_size", dist.Delta(width_size_v))

        pyro.sample("width", self.Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta).to_event(2))
                m = pyro.sample("m", dist.Categorical(m_probs[batch_idx]))
                for k in range(self.K):
                    with poutine.mask(mask=(m == k+1).byte()):
                        tril_mask = torch.ones(1,1,1,1,self.K,self.K).tril().byte()[...,k,:]
                        pyro.sample("height_{}".format(k), dist.Gamma(h_loc[batch_idx][...,k,:] * h_beta[batch_idx][...,k,:], h_beta[batch_idx][...,k,:]).mask(tril_mask).to_event(3))
                        #pyro.sample("width_{}".format(k), self.Location(w_mode[batch_idx][...,k,:], w_size[batch_idx][...,k,:], 0.5, 2.5).mask(tril_mask).to_event(3))
                        pyro.sample("x0_{}".format(k), self.Location(x_mode[batch_idx][...,k,:], size[batch_idx][...,k,:], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(3))
                        pyro.sample("y0_{}".format(k), self.Location(y_mode[batch_idx][...,k,:], size[batch_idx][...,k,:], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(3))


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
