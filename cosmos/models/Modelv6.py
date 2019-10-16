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

class Modelv6:
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, jit, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "v6"
        self.data = data
        self.dataset = dataset
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
        self.target_locs = self.target_locs.reshape(1,self.N,self.F,1,1,2).permute(1,2,3,4,0,5) # N,F,1,1,K,2
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,2,2)
        
        pyro.clear_param_store()
        self.epoch_count = 0
        self.lr = lr
        #self.prefit_optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        #self.optim = pyro.optim.Adam(per_param_args)
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["height_loc_loc", "h_loc_1", "h_beta_1", "w_loc_1", "w_beta_1", "x_mode_1", "y_mode_1", "size_1",
                                "h_loc_2", "h_beta_2", "w_loc_2", "w_beta_2", "x_mode_2", "y_mode_2", "size_2"]), self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "tracker", "{}".format(self.__name__), "K{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        k = height.shape[0]
        spot_locs = self.target_locs[batch_idx].repeat(1,1,1,1,k,1) # select target locs for given indices
        spot_locs[...,0] += x0.permute(1,2,3,4,0) # adjust for the center of the first frame
        spot_locs[...,1] += y0.permute(1,2,3,4,0) # x0 and y0 can be either scalars or vectors
        if k == 1:
            rv = dist.MultivariateNormal(spot_locs.reshape(len(batch_idx),self.F,1,1,2), scale_tril=self.spot_scale * width.reshape(len(batch_idx),self.F,1,1).view(width.size()+(1,1)))
        if k > 1:
            p = height / height.sum(dim=0)
            logits = torch.log(p/(1-p))
            logits = logits.permute(1,2,3,4,0)
            w = width.reshape(k,1,len(batch_idx),self.F,1,1).repeat(1,2,1,1,1,1).permute(2,3,4,5,0,1)
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

    
    @config_enumerate
    def model(self):
        # noise variables
        noise_params = dict()
        for var in self._params:
            noise_params[var] = pyro.sample(var, self._params[var]["prior"])

        #plates
        #K_plate = pyro.plate("K_plate", self.K-1) 
        N_plate = pyro.plate("N_plate", self.N, subsample_size=16, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        junk_pi = pyro.sample("junk_pi", dist.Dirichlet(0.5 * torch.ones(2)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1,1]))
        
        #with K_plate:
        height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
        height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
        width_loc = pyro.sample("width_loc", dist.HalfNormal(10.))
        width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
                
        x0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])
        y0_size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])

        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta))
                z = pyro.sample("z", dist.Categorical(pi))
                j = pyro.sample("j", dist.Categorical(junk_pi))
                m = z + j
                spot = {}
                for k in range(1,self.K+1):
                    with poutine.mask(mask=(m == k).byte()):
                        theta_pi = torch.ones(k)/k 
                        dist0 = dist.Multinomial(0, theta_pi)
                        dist1 = dist.Multinomial(1, theta_pi)
                        theta = pyro.sample("theta_{}".format(k), dist.MaskedMixture(z.byte(), dist0, dist1))
                        theta = theta.permute(4,0,1,2,3).long()
                        with pyro.plate("K_plate_{}".format(k), k, dim=-5):
                            height = pyro.sample("height_{}".format(k), dist.Gamma(height_loc * height_beta, height_beta))
                            width = pyro.sample("width_{}".format(k), dist.Gamma(width_loc * width_beta, width_beta))
                            x0 = pyro.sample("x0_{}".format(k), self.Location(0., x0_size[theta], -(self.D+3)/2, self.D+3))
                            y0 = pyro.sample("y0_{}".format(k), self.Location(0., y0_size[theta], -(self.D+3)/2, self.D+3))

                        spot[k] = self.gaussian_spot(batch_idx, height, width, x0, y0)

                # return locs for K classes
                locs = torch. where(m == 1, spot[1], torch.zeros_like(spot[1])) + torch.where(m == 2, spot[2], torch.zeros_like(spot[2])) + background
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
        #K_plate = pyro.plate("K_plate", self.K-1)
        N_plate = pyro.plate("N_plate", self.N, subsample_size=16, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)

        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        junk_pi_concentration = pyro.param("junk_pi_concentration", torch.ones(2)*self.N*self.F/2, constraint=constraints.positive)
        background_loc_loc = pyro.param("background_loc_loc", self.data.b_loc_1.mean()*torch.ones(1), constraint=constraints.positive)
        #background_loc_loc = pyro.param("background_loc_loc", self.data.background.mean()*torch.ones(1), constraint=constraints.positive)
        background_loc_beta = pyro.param("background_loc_beta", torch.ones(1), constraint=constraints.positive)
        background_beta_loc = pyro.param("background_beta_loc", 10*torch.ones(1), constraint=constraints.positive)
        background_beta_beta = pyro.param("background_beta_beta", torch.ones(1), constraint=constraints.positive)
        height_loc_loc = pyro.param("height_loc_loc", 1000*torch.ones(1), constraint=constraints.positive)
        height_loc_beta = pyro.param("height_loc_beta", torch.ones(1), constraint=constraints.positive)
        height_beta_loc = pyro.param("height_beta_loc", torch.ones(1)*10, constraint=constraints.positive)
        height_beta_beta = pyro.param("height_beta_beta", torch.ones(1), constraint=constraints.positive)
        #width_loc_loc = pyro.param("width_loc_loc", self.data.w_loc.mean()*torch.ones(self.K-1), constraint=constraints.positive)
        width_loc_loc = pyro.param("width_loc_loc", 1.5*torch.ones(1), constraint=constraints.positive)
        width_loc_beta = pyro.param("width_loc_beta", torch.ones(1)*10, constraint=constraints.positive)
        width_beta_loc = pyro.param("width_beta_loc", torch.ones(1)*10, constraint=constraints.positive)
        width_beta_beta = pyro.param("width_beta_beta", torch.ones(1)*10, constraint=constraints.positive)


        b_loc = pyro.param("b_loc", self.data.b_loc_1.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", self.data.b_beta_1, constraint=constraints.positive)
        
        # local variables
        w_loc, w_beta, h_loc, h_beta, x_mode, size, y_mode, theta_pi = {}, {}, {}, {}, {}, {}, {}, {}
        w_loc[1] = pyro.param("w_loc_1", self.data.w_loc_1, constraint=constraints.positive)
        w_beta[1] = pyro.param("w_beta_1", self.data.w_beta_1, constraint=constraints.positive)
        h_loc[1] = pyro.param("h_loc_1", self.data.h_loc_1.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        h_beta[1] = pyro.param("h_beta_1", self.data.h_beta_1.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        x_mode[1] = pyro.param("x_mode_1", self.data.x_mode_1.reshape(1,self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        size[1] = pyro.param("size_1", self.data.size_1.reshape(1,self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        y_mode[1] = pyro.param("y_mode_1", self.data.y_mode_1.reshape(1,self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))

        w_loc[2] = pyro.param("w_loc_2", self.data.w_loc_2, constraint=constraints.positive)
        w_beta[2] = pyro.param("w_beta_2", self.data.w_beta_2, constraint=constraints.positive)
        h_loc[2] = pyro.param("h_loc_2", self.data.h_loc_2.reshape(2,self.N,self.F,1,1), constraint=constraints.positive)
        h_beta[2] = pyro.param("h_beta_2", self.data.h_beta_2.reshape(2,self.N,self.F,1,1), constraint=constraints.positive)
        x_mode[2] = pyro.param("x_mode_2", self.data.x_mode_2.reshape(2,self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        size[2] = pyro.param("size_2", self.data.size_2.reshape(2,self.N,self.F,1,1), constraint=constraints.greater_than(2.))
        y_mode[2] = pyro.param("y_mode_2", self.data.y_mode_2.reshape(2,self.N,self.F,1,1), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))

        theta_pi[1] = pyro.param("theta_pi_1", torch.ones(self.N,self.F,1,1,1), constraint=constraints.simplex)
        theta_pi[2] = pyro.param("theta_pi_2", torch.ones(self.N,self.F,1,1,2) / 2, constraint=constraints.simplex)
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / (self.K), constraint=constraints.simplex)
        j_probs = pyro.param("j_probs", torch.ones(self.N,self.F,1,1,2) / 2, constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("junk_pi", dist.Dirichlet(junk_pi_concentration))
        pyro.sample("background_loc", dist.Gamma(background_loc_loc * background_loc_beta, background_loc_beta))
        pyro.sample("background_beta", dist.Gamma(background_beta_loc * background_beta_beta, background_beta_beta))
        #with K_plate:
        pyro.sample("height_loc", dist.Gamma(height_loc_loc * height_loc_beta, height_loc_beta))
        pyro.sample("height_beta", dist.Gamma(height_beta_loc * height_beta_beta, height_beta_beta))
        pyro.sample("width_loc", dist.Gamma(width_loc_loc * width_loc_beta, width_loc_beta))
        pyro.sample("width_beta", dist.Gamma(width_beta_loc * width_beta_beta, width_beta_beta))
        

        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("background", dist.Gamma(b_loc[batch_idx] * b_beta, b_beta))
                z = pyro.sample("z", dist.Categorical(z_probs[batch_idx]))
                j = pyro.sample("j", dist.Categorical(j_probs[batch_idx]))
                m = z + j
                for k in range(1,self.K+1):
                    with poutine.mask(mask=(m == k).byte()):
                        #dist0 = dist.Multinomial(0, theta_pi[k])
                        #dist1 = dist.Multinomial(1, theta_pi[k])
                        #theta = pyro.sample("theta_{}".format(k), dist.MaskedMixture(z.byte(), dist0, dist1))
                        with pyro.plate("K_plate_{}".format(k), k, dim=-5):
                            pyro.sample("height_{}".format(k), dist.Gamma(h_loc[k][:,batch_idx] * h_beta[k][:,batch_idx], h_beta[k][:,batch_idx]))
                            pyro.sample("width_{}".format(k), dist.Gamma(w_loc[k] * w_beta[k] * size[k][:,batch_idx], w_beta[k] * size[k][:,batch_idx]))
                            pyro.sample("x0_{}".format(k), self.Location(x_mode[k][:,batch_idx], size[k][:,batch_idx], -(self.D+3)/2, self.D+3))
                            pyro.sample("y0_{}".format(k), self.Location(y_mode[k][:,batch_idx], size[k][:,batch_idx], -(self.D+3)/2, self.D+3))

        #return height, width, background, x0, y0

    def fixed_epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.fixed.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.fixed, self.writer, feature=False)
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()


    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = self.svi.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False)
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        self.optim.save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "K{}".format(self.K), "optimizer"))
        pyro.get_param_store().save(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "K{}".format(self.K), "params"))
        np.savetxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "K{}".format(self.K), "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("Classification results were saved in {}...".format(self.data.path))

    def load(self):
        self.epoch_count = int(np.loadtxt(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "K{}".format(self.K), "epoch_count")))
        self.optim.load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "K{}".format(self.K), "optimizer"))
        pyro.get_param_store().load(os.path.join(self.data.path, "runs", "{}".format(self.dataset), "detector", 
                "{}".format(self.__name__), "K{}".format(self.K), "params"))
