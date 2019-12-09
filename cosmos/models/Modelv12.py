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
from cosmos.models.helper import Location


def per_param_args(module_name, param_name):
    if param_name in ["size", "jsize"]:
        return {"lr": 0.005}
    else:
        return {"lr": 0.002}

class Modelv12:
    """ Gaussian Spot Model """
    def __init__(self, data, dataset, K, lr, n_batch, jit, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "v12"
        self.data = data
        self.dataset = dataset
        self.N, self.F, self.D, _ = data._store.shape
        self.K = K
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
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
        pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "detector", "v11/M{}".format(self.K), "params"))
        self.epoch_count = 0
        #self.prefit_optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        #self.optim = pyro.optim.Adam(per_param_args)
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "{}".format(self.dataset), "detector", "{}".format(self.__name__), "M{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = self.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
        spot_locs[...,0] += x0 # N,F,D,D,M,K,2 .permute(1,2,3,4,5,0) # adjust for the center of the first frame
        spot_locs[...,1] += y0 #.permute(1,2,3,4,5,0) # x0 and y0 can be either scalars or vectors
        spot = torch.zeros(batch_idx.shape[0],self.F,self.D,self.D)
        for k in range(self.K):
            #w = width.reshape(1,1,1,1)
            w = width[...,k]
            rv = dist.MultivariateNormal(spot_locs[...,k,:], scale_tril=self.spot_scale * w.view(w.size()+(1,1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D,M
            spot += height[...,k] * gaussian_spot # N,F,D,D,M
        return spot
    
    @config_enumerate
    def model(self):
        # noise variables
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(0., offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(50.))

        #plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)
        
        m_pi = pyro.sample("m_pi", dist.Dirichlet(torch.ones(4) / 4))
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])
        #m_pi = pyro.sample("m_pi", dist.Dirichlet(torch.ones(3) / 3))
        height_loc = pyro.sample("height_loc", dist.HalfNormal(torch.tensor([10., 200.])).to_event(1))
        height_beta = pyro.sample("height_beta", dist.HalfNormal(torch.tensor([10., 10.])).to_event(1))

        # Global Variables
        scale = torch.tensor([10., 0.5])


        #width = pyro.sample("width", Location(torch.tensor(1.3), torch.tensor(10.), 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.HalfNormal(1000.))
                m = pyro.sample("m", dist.Categorical(m_pi)) # N,F,1,1
                m = m_matrix[m] # N,F,1,1,K
                height = pyro.sample("height", dist.Gamma(height_loc[m] * height_beta[m], height_beta[m]).to_event(1)) # K,N,F,1,1
                width = pyro.sample("width", Location(1.3, 10., 0.5, 2.5).expand([1,1,1,1,self.K]).to_event(1))
                x0 = pyro.sample("x0", dist.Normal(torch.tensor(0.), 10.).expand([1,1,1,1,self.K]).to_event(1))
                y0 = pyro.sample("y0", dist.Normal(torch.tensor(0.), 10.).expand([1,1,1,1,self.K]).to_event(1))

                locs = self.gaussian_spot(batch_idx, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, gain, offset), obs=self.data[batch_idx])
    
    @config_enumerate
    def guide(self):
        offset_max = self.data._store.min() - 0.1
        offset_v = pyro.param("offset_v", offset_max-50, constraint=constraints.interval(0.,offset_max))
        gain_v = pyro.param("gain_v", torch.tensor(5.), constraint=constraints.positive)
        pyro.sample("offset", dist.Delta(offset_v))
        pyro.sample("gain", dist.Delta(gain_v))

        # plates
        N_plate = pyro.plate("N_plate", self.N, subsample_size=self.n_batch, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, dim=-3)

        # Global Parameters
        b_loc = pyro.param("b_loc")
        w_mode, w_size= pyro.param("w_mode"), pyro.param("w_size")
        h_loc = pyro.param("h_loc")
        x_mode, y_mode = pyro.param("x_mode"), pyro.param("y_mode")
        
        height_loc_v = pyro.param("height_loc_v", torch.tensor([5., 200.]), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.tensor([1., 1.]), constraint=constraints.positive)

        ######
        h_max = np.percentile(pyro.param("h_loc").detach().cpu(), 95)
        p = torch.where(pyro.param("h_loc").detach() < h_max, pyro.param("h_loc").detach()/h_max, torch.tensor(1.))

        #m2 = p * 0.45 + 0.025
        #m1 = p * 0.45 + 0.025
        #m0 = 1 - m1 - m2
        m1 = p * 0.9 + 0.05
        m0 = 1 - m1
        #m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])

        #m_probs = torch.stack((m0,m1), dim=-1)
        #m_probs = pyro.param("m_probs", m_probs.reshape(self.N,self.F,1,1,4), constraint=constraints.simplex)
        m_probs = pyro.param("m_probs", torch.ones(self.N,self.F,1,1,4), constraint=constraints.simplex)
        m_pi_concentration = pyro.param("m_pi_concentration", torch.ones(4)*self.N*self.F/4, constraint=constraints.positive)

        pyro.sample("m_pi", dist.Dirichlet(m_pi_concentration))
        pyro.sample("height_loc", dist.Delta(height_loc_v).to_event(1))
        pyro.sample("height_beta", dist.Delta(height_beta_v).to_event(1))
        #width = pyro.sample("width", Location(w_mode, w_size, 0.5, 2.5))
        with N_plate as batch_idx:
            with F_plate:
                # AoI & Frame Local Variables
                background = pyro.sample("background", dist.Gamma(b_loc[batch_idx] * self.D**2, self.D**2))
                pyro.sample("m", dist.Categorical(m_probs[batch_idx]))
                height = pyro.sample("height", dist.Gamma(h_loc[batch_idx], 1.).to_event(1))
                width = pyro.sample("width", Location(w_mode[batch_idx], w_size[batch_idx], 0.5, 2.5).to_event(1))
                scale = torch.sqrt((width**2 + 1/12) / height + 8 * math.pi * width**4 * background.unsqueeze(dim=-1) / height**2)
                pyro.sample("x0", dist.Normal(x_mode[batch_idx], scale).to_event(1))
                pyro.sample("y0", dist.Normal(y_mode[batch_idx], scale).to_event(1))
                #pyro.sample("x0", Location(x_mode[:,batch_idx], size[:,batch_idx], -(self.D+3)/2, self.D+3))
                #pyro.sample("y0", Location(y_mode[:,batch_idx], size[:,batch_idx], -(self.D+3)/2, self.D+3))


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
            #with torch.autograd.detect_anomaly():
            epoch_loss = self.svi.step()
            #torch.save(pyro.param("size"), os.path.join(self.data.path, "runs", "size{}".format(epoch)))
            #torch.save(pyro.param("x_mode"), os.path.join(self.data.path, "runs", "xmode{}".format(epoch)))
            #torch.save(pyro.param("y_mode"), os.path.join(self.data.path, "runs", "ymode{}".format(epoch)))
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
