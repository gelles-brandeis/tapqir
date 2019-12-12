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

from scipy.stats import bernoulli, poisson

def per_param_args(module_name, param_name):
    if param_name in ["size", "jsize"]:
        return {"lr": 0.002}
    else:
        return {"lr": 0.0005}

def m_param(pi, lamda, K):
    m_pi = torch.zeros(K+1)
    m_pi[0] = bernoulli(0, pi) * poisson.pmf(0, lamda)
    for k in range(1,K+1):
        m_pi[k] = bernoulli.pmf(1, pi) * poisson.pmf(k-1, lamda) + bernoulli.pmf(0, pi) * poisson(k, lamda)
    return m_pi

def theta_param(pi, lamda, K):
    theta_pi = torch.zeros(1,1,1,1,1,K+1,K+1)
    theta_pi[...,0,0] = 1
    for k in range(1,K+1):
        theta_pi[...,k,0] = bernoulli(0, pi) * poisson(k, lamda) / m_param(pi, lamda, K)[k]
        theta_pi[...,k,1:k+1] = bernoulli(1, pi) * poisson(k-1, lamda) / m_param(pi, lamda, K)[k]
    return theta_pi

def theta_encoder(theta_cat, K, nidx, fidx, xidx, yidx):
    theta = torch.zeros(theta_cat.size() + (1,1,K+1,))
    theta[nidx,fidx,xidx,yidx,theta_cat] = 1
    return theta[...,1:].long()
    
def m_mask(m, K):
    mask = torch.zeros(m.size()+(1,1,K))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            mask[i,j,0,0,:m[i,j]+1] = 1
    return mask

class Modelv7:
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, n_batch, jit, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "v7"
        self.data = data
        self.dataset = dataset
        self.N, self.F, self.D, _ = data._store.shape
        self.K = K
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        self.n_batch = n_batch
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).to(torch.float32)
        self.pixel_pos = self.pixel_pos.reshape(1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(self.N,self.F,1,1,1,1,2).repeat(1,1,1,1,self.K,self.K,1)# N,F,1,1,M,K,2
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)
        
        pyro.clear_param_store()
        #pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "junk", "M2.noamort", "params"))
        pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "junk", "M2/lr/0.0005/h/10", "params"))
        self.epoch_count = 0
        self.lr = lr
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        #self.optim = pyro.optim.Adam(per_param_args)
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["height_loc_loc", "h_loc", "h_beta", "w_loc", "w_beta", "x_mode", "y_mode", "size"]), 
        #self.fixed = SVI(self.model, poutine.block(self.guide, hide=["height_loc_loc"]), 
                                        self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "tracker", "{}".format(self.__name__), "M{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = self.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
        spot_locs[...,0] += x0 # N,F,D,D,M,K,2 .permute(1,2,3,4,5,0) # adjust for the center of the first frame
        spot_locs[...,1] += y0 #.permute(1,2,3,4,5,0) # x0 and y0 can be either scalars or vectors
        # N,F,D,D,M,K -> tril
        #height[...,1,:] = 0
        spot = torch.zeros(len(batch_idx),self.F,self.D,self.D,self.K+1)
        for k in range(self.K):
            if k == 0:
                #spot_locs = spot_locs.reshape(len(batch_idx),self.F,1,1,2) 
                width = width.reshape(1,1,1,1)
                rv = dist.MultivariateNormal(spot_locs[...,0,0,:], scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
                gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D,M
                spot[...,k+1] = height[...,k,k] * gaussian_spot # N,F,D,D,M
            else:
                #height = height[...,self.M-1,:self.M]
                p = height[...,k,:k+1] / height[...,k,:k+1].sum(dim=-1, keepdims=True)
                logits = torch.log(p/(1-p))
                #logits = logits # N,F,D,D,M,K
                #w = width.unsqueeze(dim=-1).repeat(1,1,1,1,1,1,2) # N,F,D,D,M,K,2
                #w = width.reshape(1,1,1,1,1,1,1).repeat(len(batch_idx),self.F,1,1,2,2,2) # N,F,D,D,M,K,2
                w = width.reshape(1,1,1,1,1,1).repeat(len(batch_idx),self.F,1,1,k+1,2) # N,F,D,D,M,K,2
                rv = dist.MixtureOfDiagNormals(locs=spot_locs[...,k,:k+1,:], coord_scale=w, component_logits=logits)
                gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D,M
                spot[...,k+1] = height[...,k,:k+1].sum(dim=-1) * gaussian_spot # N,F,D,D,M
        return spot
    
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
        #noise_params = dict()
        #for var in self._params:
        #    noise_params[var] = pyro.sample(var, self._params[var]["prior"])
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(torch.tensor(0.), offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(torch.tensor(50.)))

        #plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-2)
        F_plate = pyro.plate("F_plate", self.F, dim=-1)
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        lamda = pyro.sample("lamda", dist.Gamma(0.01 * torch.ones(1), torch.tensor(0.1)))
        m_pi = m_param(pi, lamda, self.K)
        theta_pi = theta_param(pi, lamda, self.K)
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.))
        height_loc = pyro.sample("height_loc", dist.HalfNormal(3000.))
        height_beta = pyro.sample("height_beta", dist.HalfNormal(500.))
        #width_mode = pyro.sample("width_mode", self.Location(1.3, 4., 0.5, 2.5))
        #width_size = pyro.sample("width_size", dist.HalfNormal(500.))
        x0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])
        y0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])

        #width = pyro.sample("width", self.Location(1.3, 4., 0.5, 2.5))
        with N_plate as batch_idx:
            nind, find, xind, yind = torch.meshgrid(torch.arange(len(batch_idx)),torch.arange(self.F),torch.arange(self.D),torch.arange(self.D))
            nidx, fidx, xidx, yidx = torch.meshgrid(torch.arange(len(batch_idx)),torch.arange(self.F),torch.arange(1),torch.arange(1))
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta).expand([1,1,1,1]).to_event(2))
                m = pyro.sample("m", dist.Categorical(m_pi))
                theta_cat = pyro.sample("theta", dist.Categorical(theta_pi[m]))
                theta = theta_encoder(theta_cat, self.K, nidx, fidx, xidx, yidx)
                #z = pyro.sample("z", dist.Categorical(pi))
                #j = pyro.sample("j", dist.Categorical(junk_pi))
                #m = z + j
                #tril_mask = torch.ones(1,1,1,1,self.K,self.K).tril().byte()
                mask = m_mask(m, self.K)
                #with poutine.mask(mask=z.byte()):
                #    theta_pi = torch.ones(1,1,1,1,self.K,self.K).tril() 
                #    theta = pyro.sample("theta", dist.OneHotCategorical(theta_pi).to_event(3)) # N,F,1,1,M,K
                #    theta = theta.masked_fill((1-z.view(z.size()+(1,1,1,1))).bool(), 0).long()
                #with poutine.mask(mask=(m > 0).byte()):
                height = pyro.sample("height", dist.Gamma(height_loc * height_beta, height_beta).expand([1,1,1,1,self.K]).mask(mask).to_event(3))
                width = pyro.sample("width", self.Location(1.3, 4., 0.5, 2.5))
                x0 = pyro.sample("x0", self.Location(0., x0_size[theta], -(self.D+3)/2, self.D+3).mask(mask).to_event(2))
                y0 = pyro.sample("y0", self.Location(0., y0_size[theta], -(self.D+3)/2, self.D+3).mask(mask).to_event(2))

                spot = self.gaussian_spot(batch_idx, height, width, x0, y0)
                #spot = torch.cat([torch.zeros(len(batch_idx),self.F,self.D,self.D,1), spot], dim=-1)
                #locs = spot[nind,find,xind,yind,m.view(m.size()+(1,1))] + background
                locs = spot + background
                pyro.sample("data", self.CameraUnit(locs, gain, offset).to_event(2), obs=self.data[batch_idx])
    
    @config_enumerate
    def guide(self):
        offset_v = pyro.param("offset_v")
        gain_v = pyro.param("gain_v")
        pyro.sample("offset", dist.Delta(offset_v))
        pyro.sample("gain", dist.Delta(gain_v))

        h1_max = np.percentile(pyro.param("h_loc").detach().cpu()[...,1,0], 95)
        p1 = torch.where(pyro.param("h_loc").detach()[...,1,0] < h1_max, pyro.param("h_loc").detach()[...,1,0]/h1_max, torch.tensor(1.))
        h2_max = np.percentile(pyro.param("h_loc").detach().cpu()[...,1,1], 95)
        p2 = torch.where(pyro.param("h_loc").detach()[...,1,1] < h2_max, pyro.param("h_loc").detach()[...,1,1]/h2_max, torch.tensor(1.))

        #r = torch.sqrt(pyro.param("x_mode").detach() ** 2 + pyro.param("y_mode").detach() ** 2)
        #r = 1 - r / r.sum(dim=-1, keepdims=True)
        #j = torch.where(r < 2, r/2, torch.tensor(1.))

        #z1 = p1 * (1-j[...,0,0]) * 0.9 + 0.05
        z1 = p2 * 0.9 + 0.05
        z0 = 1 - z1
        j1 = p1 * 0.9 + 0.05
        j0 = 1 - j1

        z_probs = torch.stack((z0,z1), dim=-1)
        j_probs = torch.stack((j0,j1), dim=-1)

        # plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-2)
        F_plate = pyro.plate("F_plate", self.F, dim=-1)

        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", torch.ones(self.K)*self.N*self.F/(self.K), constraint=constraints.positive)
        lamda_loc = pyro.param("lamda_loc", torch.ones(1)*0.1, constraint=constraints.positive)
        lamda_beta = pyro.param("lamda_beta", torch.ones(1)*10., constraint=constraints.positive)
        background_loc_loc = pyro.param("background_loc_loc", self.data.background.mean(), constraint=constraints.positive)
        background_loc_beta = pyro.param("background_loc_beta", torch.ones(1)*500, constraint=constraints.positive)
        background_beta_loc = pyro.param("background_beta_loc", torch.ones(1), constraint=constraints.positive)
        background_beta_beta = pyro.param("background_beta_beta", torch.ones(1)*500, constraint=constraints.positive)
        height_loc_loc = pyro.param("height_loc_loc", 1000*torch.ones(1), constraint=constraints.positive)
        height_loc_beta = pyro.param("height_loc_beta", torch.ones(1)*100, constraint=constraints.positive)
        height_beta_loc = pyro.param("height_beta_loc", torch.ones(1), constraint=constraints.positive)
        height_beta_beta = pyro.param("height_beta_beta", torch.ones(1)*500, constraint=constraints.positive)

        b_loc, b_beta= pyro.param("b_loc"), pyro.param("b_beta")
        w_mode, w_size= pyro.param("w_mode"), pyro.param("w_size")
        h_loc, h_beta= pyro.param("h_loc"), pyro.param("h_beta")
        x_mode, y_mode, size= pyro.param("x_mode"), pyro.param("y_mode"), pyro.param("size")

        theta_probs = torch.ones(self.N,self.F,1,1,self.K+1,self.K+1).tril()
        #theta_probs[...,1,0] = 0.2
        #theta_probs[...,1,1] = 0.8
        theta_probs = pyro.param("theta_probs", theta_probs, constraint=constraints.simplex)
        m_probs = pyro.param("m_probs", torch.ones(self.N,self.F,self.K+1)/3, constraint=constraints.simplex)
        #j_probs = pyro.param("j_probs", torch.ones(self.N,self.F,2)/2, constraint=constraints.simplex)
        #z_probs = pyro.param("z_probs", z_probs.reshape(self.N,self.F,self.K), constraint=constraints.simplex)
        #j_probs = pyro.param("j_probs", j_probs.reshape(self.N,self.F,2), constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("lamda", dist.Gamma(lamda_loc * lamda_beta, lamda_beta))
        pyro.sample("background_loc", dist.Gamma(background_loc_loc * background_loc_beta, background_loc_beta))
        pyro.sample("background_beta", dist.Gamma(background_beta_loc * background_beta_beta, background_beta_beta))
        pyro.sample("height_loc", dist.Gamma(height_loc_loc * height_loc_beta, height_loc_beta))
        pyro.sample("height_beta", dist.Gamma(height_beta_loc * height_beta_beta, height_beta_beta))
        
        pyro.sample("width", self.Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta).to_event(2))
                m = pyro.sample("m", dist.Categorical(m_probs[batch_idx]))
                pyro.sample("theta", dist.Categorical(theta_probs[batch_idx]).to_event(3)) # N,F,1,1,M,K
                tril_mask = torch.ones(1,1,1,1,self.K,self.K).tril().byte()
                with poutine.mask(mask=(m > 0).byte()):
                    pyro.sample("height", dist.Gamma(h_loc[batch_idx] * h_beta[batch_idx], h_beta[batch_idx]).mask(tril_mask).to_event(4))
                    pyro.sample("x0", self.Location(x_mode[batch_idx], size[batch_idx], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4)) # N,F,1,1,M,K
                    pyro.sample("y0", self.Location(y_mode[batch_idx], size[batch_idx], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4))

                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta).expand([1,1,1,1]).to_event(2))
                m = pyro.sample("m", dist.Categorical(m_pi))
                theta_cat = pyro.sample("theta", dist.Categorical(theta_pi[m]))
                theta = theta_encoder(theta_cat, self.K, nidx, fidx, xidx, yidx)
                #z = pyro.sample("z", dist.Categorical(pi))
                #j = pyro.sample("j", dist.Categorical(junk_pi))
                #m = z + j
                tril_mask = torch.ones(1,1,1,1,self.K,self.K).tril().byte()
                #with poutine.mask(mask=z.byte()):
                #    theta_pi = torch.ones(1,1,1,1,self.K,self.K).tril() 
                #    theta = pyro.sample("theta", dist.OneHotCategorical(theta_pi).to_event(3)) # N,F,1,1,M,K
                #    theta = theta.masked_fill((1-z.view(z.size()+(1,1,1,1))).bool(), 0).long()
                with poutine.mask(mask=(m > 0).byte()):
                    height = pyro.sample("height", dist.Gamma(height_loc * height_beta, height_beta).expand([1,1,1,1,self.K,self.K]).mask(tril_mask).to_event(4))
                    x0 = pyro.sample("x0", self.Location(0., x0_size[theta], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4))
                    y0 = pyro.sample("y0", self.Location(0., y0_size[theta], -(self.D+3)/2, self.D+3).mask(tril_mask).to_event(4))


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
        self.optim.save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "tracker", 
                "{}".format(self.__name__), "M{}".format(self.K), "optimizer"))
        pyro.get_param_store().save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "tracker", 
                "{}".format(self.__name__), "M{}".format(self.K), "params"))
        np.savetxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "tracker", 
                "{}".format(self.__name__), "M{}".format(self.K), "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("Classification results were saved in {}...".format(self.data.path))

    def load(self):
        try:
            self.epoch_count = int(np.loadtxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "tracker", 
                    "{}".format(self.__name__), "M{}".format(self.K), "epoch_count")))
            self.optim.load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "tracker", 
                    "{}".format(self.__name__), "M{}".format(self.K), "optimizer"))
            pyro.get_param_store().load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "tracker", 
                    "{}".format(self.__name__), "M{}".format(self.K), "params"))
            print("loaded previous run")
        except:
            pass
