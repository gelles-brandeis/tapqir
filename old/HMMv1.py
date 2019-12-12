import math
import os
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine
import pyro.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import write_summary
from glimpse_reader import Sampler
from tqdm import tqdm
from models.noise import _noise, _noise_fn


class HMMv1:
    """ Gaussian Spot Model """
    def __init__(self, data, K, lr, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.__name__ = "v1"
        self.data = data
        self.N, self.F, self.D, _ = data._store.shape
        assert K >= 2
        self.K = K
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).float()
        self.pixel_pos = self.pixel_pos.reshape(1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(self.N,self.F,1,1,2).repeat(1,1,1,1,1)
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,2,2)
        
        pyro.clear_param_store()
        self.epoch_count = 0
        self.lr = lr
        #self.prefit_optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = TraceEnum_ELBO()
        self.fixed = SVI(self.model, poutine.block(self.guide, hide=["h_loc", "h_beta", "b_loc", "b_beta",
            "w_loc", "w_beta", "x_loc", "x_scale", "y_loc", "y_scale", "x0_scale_v", "y0_scale_v", "height_loc_v"]), self.optim, loss=self.elbo) 
        self.svi = SVI(self.model, poutine.block(self.guide, hide=["x0_scale_v", "y0_scale_v"]), self.optim, loss=self.elbo) 
        #self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        #self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "classifier", "K{}".format(self.K), "lr{}".format(self.lr)))
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "hmm", "{}".format(self.__name__), "K{}".format(self.K)))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, frame, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        # select target locs for given indices
        spot_locs = self.target_locs[batch_idx,frame] # ind,F,D,D,2
        # adjust for the center of the first frame
        # x0 and y0 can be either scalars or vectors
        spot_locs[...,0] += x0
        spot_locs[...,1] += y0
        rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        # height can be either a scalar or a vector
        return height * gaussian_spot #
    
    @config_enumerate
    def model(self, batch_idx):
        # noise variables
        noise_params = dict()
        for var in self._params:
            noise_params[var] = pyro.sample(var, self._params[var]["prior"])

        #plates
        K_plate = pyro.plate("K_plate", self.K-1)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-3)
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        trans = pyro.sample("trans", dist.Dirichlet(0.9 * torch.eye(self.K) + 0.1).to_event(1))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1]))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(10.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(10.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(10.))

        height_loc = torch.cat((torch.ones(1), height_loc), 0)
        height_beta = torch.cat((torch.ones(1), height_beta), 0)
        width_loc = torch.cat((torch.ones(1), width_loc), 0)
        width_beta = torch.cat((torch.ones(1), width_beta), 0)
        x0_scale = torch.cat((torch.ones(1), x0_scale), 0)
        y0_scale = torch.cat((torch.ones(1), y0_scale), 0)
                
        with N_plate:
            for frame in pyro.markov(range(self.F)):
                if frame == 0:
                    z = pyro.sample("z_{}".format(frame), dist.Categorical(pi))
                else:
                    z = pyro.sample("z_{}".format(frame), dist.Categorical(trans[z]))
                background = pyro.sample("background_{}".format(frame), dist.Gamma(background_loc*background_beta, background_beta))
                # AoI & Frame Local Variables
                with poutine.mask(mask=(z > 0).byte()):
                    height = pyro.sample("height_{}".format(frame), dist.Gamma(height_loc[z] * height_beta[z], height_beta[z]))
                    width = pyro.sample("width_{}".format(frame), dist.Gamma(width_loc[z] * width_beta[z], width_beta[z]))
                    x0 = pyro.sample("x0_{}".format(frame), dist.Normal(0., x0_scale[z]))
                    y0 = pyro.sample("y0_{}".format(frame), dist.Normal(0., y0_scale[z]))
            
                # return locs for K classes
                spots = self.gaussian_spot(batch_idx, frame, height, width, x0, y0)
                locs = torch.where(z > 0, spots, torch.zeros_like(spots))
                locs += background
                with pyro.plate("x_plate_{}".format(frame), size=self.D, dim=-2):
                    with pyro.plate("y_plate_{}".format(frame), size=self.D, dim=-1):
                        pyro.sample("data_{}".format(frame), self.CameraUnit(locs, **noise_params), obs=self.data[batch_idx,frame])
    
    @config_enumerate
    def guide(self, batch_idx):
        # noise variables
        for var in self._params:
            guide_params = dict()
            for param in self._params[var]["guide_params"]:
                guide_params[param] = pyro.param(**self._params[var]["guide_params"][param]) 
            pyro.sample(var, self._params[var]["guide_dist"](**guide_params))

        # plates
        K_plate = pyro.plate("K_plate", self.K-1)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-3)
        
        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        trans_concentration = pyro.param("trans_concentration", torch.eye(self.K)*self.N/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.b.mean(), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", 100*torch.ones(self.K-1), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K-1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", self.data.w.mean()*torch.ones(self.K-1), constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1), constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", torch.ones(self.K-1), constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", torch.ones(self.K-1), constraint=constraints.positive)
        #x0_scale_v = pyro.param("x0_scale_v", torch.tensor([10., 1.]), constraint=constraints.positive)
        #y0_scale_v = pyro.param("y0_scale_v", torch.tensor([10., 1.]), constraint=constraints.positive)
        
        # AoI & Frame Local Parameters
        b_loc = pyro.param("b_loc", self.data.b.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", self.data.b_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        h_loc = pyro.param("h_loc", self.data.h.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", self.data.h_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        w_loc = pyro.param("w_loc", self.data.w.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        w_beta = pyro.param("w_beta", self.data.w_beta.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        x_loc = pyro.param("x_loc", self.data.x0.reshape(self.N,self.F,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", self.data.x0_scale.reshape(self.N,self.F,1,1), constraint=constraints.positive)
        y_loc = pyro.param("y_loc", self.data.y0.reshape(self.N,self.F,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", self.data.y0_scale.reshape(self.N,self.F,1,1), constraint=constraints.positive)

        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        #z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("trans", dist.Dirichlet(trans_concentration).to_event(1))
        pyro.sample("background_loc", dist.Delta(background_loc_v))
        pyro.sample("background_beta", dist.Delta(background_beta_v))
        with K_plate:
                pyro.sample("height_loc", dist.Delta(height_loc_v))
                pyro.sample("height_beta", dist.Delta(height_beta_v))
                pyro.sample("width_loc", dist.Delta(width_loc_v))
                pyro.sample("width_beta", dist.Delta(width_beta_v))
                pyro.sample("x0_scale", dist.Delta(x0_scale_v))
                pyro.sample("y0_scale", dist.Delta(y0_scale_v))

        
        with N_plate:
            for frame in pyro.markov(range(self.F)):
                # AoI Local Variables
                if frame == 0:
                    z = pyro.sample("z_{}".format(frame), dist.Categorical(z_probs[batch_idx,frame]))
                else:
                    z = pyro.sample("z_{}".format(frame), dist.Categorical(z_probs[batch_idx,frame]))
                pyro.sample("background_{}".format(frame), dist.Gamma(b_loc[batch_idx,frame]*b_beta[batch_idx,frame], b_beta[batch_idx,frame]))
                with poutine.mask(mask=(z > 0).byte()):
                    pyro.sample("height_{}".format(frame), dist.Gamma(h_loc[batch_idx,frame]*h_beta[batch_idx,frame], h_beta[batch_idx,frame]))
                    pyro.sample("width_{}".format(frame), dist.Gamma(w_loc[batch_idx,frame]*w_beta[batch_idx,frame], w_beta[batch_idx,frame]))
                    pyro.sample("x0_{}".format(frame), dist.Normal(x_loc[batch_idx,frame], x_scale[batch_idx,frame]))
                    pyro.sample("y0_{}".format(frame), dist.Normal(y_loc[batch_idx,frame], y_scale[batch_idx,frame]))
    
    def fixed_epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                loss = self.fixed.step(batch_idx)
                epoch_loss += loss * len(batch_idx) / self.N
            if not (self.epoch_count % 5):    
                write_summary(self.epoch_count, epoch_loss, self, self.fixed, self.writer, feature=False, mcc=False)
            self.epoch_count += 1

    def epoch(self, n_batch, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for batch_idx in DataLoader(Sampler(self.N), batch_size=n_batch, shuffle=True):
                loss = self.svi.step(batch_idx)
                epoch_loss += loss * len(batch_idx) / self.N
            if not (self.epoch_count % 5):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False, mcc=False)
            self.epoch_count += 1

    def save(self):
        for p in pyro.get_param_store().get_all_param_names():
            torch.save(pyro.param(p).detach().squeeze(), os.path.join(self.data.path, "runs", "classifier", 
                "{}".format(self.__name__), "K{}".format(self.K), "{}.pt".format(p)))
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "classifier", "{}".format(self.__name__), "K{}".format(self.K)))
        print("Classification results were saved in {}...".format(self.data.path))
        
