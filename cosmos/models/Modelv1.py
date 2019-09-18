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


class Modelv1:
    """ Gaussian Spot Model """
    def __init__(self, data, K, lr):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.data = data
        self.N, self.F, self.D, _ = data._store.shape
        self.K = K
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).float()
        self.pixel_pos = self.pixel_pos.reshape(1,1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(1,self.N,self.F,1,1,2).repeat(self.K-1,1,1,1,1,1)
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,1,2,2)
        
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
        self.writer = SummaryWriter(log_dir=os.path.join(self.data.path,"runs", "classifier", "K{}".format(self.K), "model1"))
        
    # Ideal 2D gaussian spot
    def gaussian_spot(self, batch_idx, frame_idx, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        # select target locs for given indices
        spot_locs = self.target_locs[:,batch_idx][:,:,frame_idx] # ind,F,D,D,2
        # adjust for the center of the first frame
        # x0 and y0 can be either scalars or vectors
        spot_locs[...,0] += x0
        spot_locs[...,1] += y0
        #print(spot_locs.shape, self.spot_scale[ind].shape, width.reshape(-1,1,1,1,1,1).shape)
        rv = dist.MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        #print(rv.batch_shape,rv.event_shape)
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        # height can be either a scalar or a vector
        # 1,K,ind,F,D,D
        return height * gaussian_spot #
    
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        #plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.meshgrid(torch.arange(len(batch_idx)),torch.arange(len(frame_idx)),torch.arange(self.D),torch.arange(self.D))
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1,1,1]))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(10.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(10.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(10.))
                
        
        with N_plate:
            # AoI Local Variables
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(pi))
                background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta))
                with poutine.mask(mask=z.view((1,)+z.size()).byte()):
                    height = pyro.sample("height", dist.Gamma(height_loc*height_beta, height_beta))
                    width = pyro.sample("width", dist.Gamma(width_loc*width_beta, width_beta))
                    x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                    y0 = pyro.sample("y0", dist.Normal(0., y0_scale))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind], N_plate, F_plate
    
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        # plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
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
                100*torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", 
                torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", 
                self.data.w.mean()*torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", 
                torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        
        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.b.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", self.data.b_beta.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        h_loc = pyro.param("h_loc", self.data.h.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", self.data.h_beta.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        w_loc = pyro.param("w_loc", self.data.w.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        w_beta = pyro.param("w_beta", self.data.w_beta.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        #x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,self.F,1,1), constraint=constraints.real)
        #x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,self.F,1,1), constraint=constraints.positive)
        #y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,self.F,1,1), constraint=constraints.real)
        #y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,self.F,1,1), constraint=constraints.positive)
        x_loc = pyro.param("x_loc", self.data.x0.reshape(1,self.N,self.F,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", self.data.x0_scale.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        y_loc = pyro.param("y_loc", self.data.y0.reshape(1,self.N,self.F,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", self.data.y0_scale.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        
        # AoI & Frame Local Parameters
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        #z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)
        
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
                
        
        with N_plate:
            # AoI Local Variables
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                pyro.sample("background", dist.Gamma(b_loc[:,batch_idx][:,:,frame_idx]*b_beta[:,batch_idx][:,:,frame_idx], b_beta[:,batch_idx][:,:,frame_idx]))
                with poutine.mask(mask=z.view((1,)+z.size()).byte()):
                    pyro.sample("height", dist.Gamma(h_loc[:,batch_idx][:,:,frame_idx]*h_beta[:,batch_idx][:,:,frame_idx], h_beta[:,batch_idx][:,:,frame_idx]))
                    pyro.sample("width", dist.Gamma(w_loc[:,batch_idx][:,:,frame_idx]*w_beta[:,batch_idx][:,:,frame_idx], w_beta[:,batch_idx][:,:,frame_idx]))
                    pyro.sample("x0", dist.Normal(x_loc[:,batch_idx][:,:,frame_idx], x_scale[:,batch_idx][:,:,frame_idx]))
                    pyro.sample("y0", dist.Normal(y_loc[:,batch_idx][:,:,frame_idx], y_scale[:,batch_idx][:,:,frame_idx]))
        
        #return height, width, background, x0, y0

    
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
            "K{}".format(self.K), "model1", "probs.pt"))
        print("Classification results were saved in {}...".format(self.data.path))
        
class Modelv1p1(Modelv1):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        #plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        #nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))
        nind, find, xind, yind = torch.meshgrid(torch.arange(len(batch_idx)),torch.arange(len(frame_idx)),torch.arange(self.D),torch.arange(self.D))
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1,1,1]))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(10.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(10.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(10.))
            with N_plate:
                height = pyro.sample("height", dist.Gamma(height_loc*height_beta, height_beta))
                width = pyro.sample("width", dist.Gamma(width_loc*width_beta, width_beta))
                x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                y0 = pyro.sample("y0", dist.Normal(0., y0_scale))
                
        
        with N_plate:
            # AoI Local Variables
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(pi))
                background = pyro.sample("background", dist.Gamma(background_loc*background_beta, background_beta))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind], N_plate, F_plate
    
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        # plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
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
                self.data.h.mean()*torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", 
                torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", 
                self.data.w.mean()*torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", 
                torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        
        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.b.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", self.data.b_beta.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        h_loc = pyro.param("h_loc", self.data.h.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", self.data.h_beta.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        w_loc = pyro.param("w_loc", self.data.w.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        w_beta = pyro.param("w_beta", self.data.w_beta.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        
        # AoI & Frame Local Parameters
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        #z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)
        
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
            with N_plate:
                pyro.sample("height", dist.Gamma(h_loc[:,batch_idx]*h_beta[:,batch_idx], h_beta[:,batch_idx]))
                pyro.sample("width", dist.Gamma(w_loc[:,batch_idx]*w_beta[:,batch_idx], w_beta[:,batch_idx]))
                pyro.sample("x0", dist.Normal(x_loc[:,batch_idx], x_scale[:,batch_idx]))
                pyro.sample("y0", dist.Normal(y_loc[:,batch_idx], y_scale[:,batch_idx]))
                
        
        with N_plate:
            # AoI Local Variables
            
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                pyro.sample("background", dist.Gamma(b_loc[:,batch_idx][:,:,frame_idx]*b_beta[:,batch_idx][:,:,frame_idx], b_beta[:,batch_idx][:,:,frame_idx]))
                #background = pyro.sample("background", dist.Delta(background_v[:,batch_idx][:,:,frame_idx]))


class Modelv1p2(Modelv1p1):
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        # plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        
        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", 
                torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", 
                self.data.background.mean(), constraint=constraints.positive)
        background_scale_v = pyro.param("background_scale_v", 
                torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", 
                self.data.height.mean()*torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        height_scale_v = pyro.param("height_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_alpha_v = pyro.param("width_alpha_v", 
                torch.ones(self.K-1,1,1,1,1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", 
                torch.ones(self.K-1,1,1,1,1)*0.1, constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        
        # AoI Local Parameters
        b_v = pyro.param("b_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        h_alpha = pyro.param("h_alpha", self.data.height.mean()*torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        w_v = pyro.param("w_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        
        # AoI & Frame Local Parameters
        #z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("background_loc", dist.Delta(background_loc_v))
        pyro.sample("background_scale", dist.Delta(background_scale_v))
        with K_plate:
            pyro.sample("height_loc", dist.Delta(height_loc_v))
            pyro.sample("height_scale", dist.Delta(height_scale_v))
            pyro.sample("width_alpha", dist.Delta(width_alpha_v))
            pyro.sample("width_beta", dist.Delta(width_beta_v))
            pyro.sample("x0_scale", dist.Delta(x0_scale_v))
            pyro.sample("y0_scale", dist.Delta(y0_scale_v))
            with N_plate:
                pyro.sample("height", dist.Gamma(h_alpha[:,batch_idx], h_beta[:,batch_idx]))
                pyro.sample("width", dist.Delta(w_v[:,batch_idx]))
                pyro.sample("x0", dist.Normal(x_loc[:,batch_idx], x_scale[:,batch_idx]))
                pyro.sample("y0", dist.Normal(y_loc[:,batch_idx], y_scale[:,batch_idx]))
                
        
        with N_plate:
            # AoI Local Variables
            pyro.sample("background", dist.Delta(b_v[:,batch_idx]))
            
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                #background = pyro.sample("background", dist.Delta(background_v[:,batch_idx][:,:,frame_idx]))
 

class Modelv1p3(Modelv1):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        #plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        #nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))
        nind, find, xind, yind = torch.meshgrid(torch.arange(len(batch_idx)),torch.arange(len(frame_idx)),torch.arange(self.D),torch.arange(self.D))
        
        # Global Variables
        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_scale = pyro.sample("background_scale", dist.HalfNormal(100.).expand([1,1,1,1,1]))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_scale = pyro.sample("height_scale", dist.HalfNormal(50.))
            width_alpha = pyro.sample("width_alpha", dist.HalfNormal(50.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(5.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(5.))
            with N_plate:
                height = pyro.sample("height", dist.Normal(height_loc, height_scale))
                width = pyro.sample("width", dist.Gamma(width_alpha, width_beta))
                with F_plate: 
                    x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                    y0 = pyro.sample("y0", dist.Normal(0., y0_scale))
                
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(pi))
                #background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind], N_plate, F_plate
 
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        # plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        
        # Global Parameters
        pi_concentration = pyro.param("pi_concentration", 
                torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", 
                self.data.background.mean(), constraint=constraints.positive)
        background_scale_v = pyro.param("background_scale_v", 
                torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", 
                self.data.height.mean()*torch.ones(self.K-1,1,1,1,1), constraint=constraints.positive)
        height_scale_v = pyro.param("height_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_alpha_v = pyro.param("width_alpha_v", 
                torch.ones(self.K-1,1,1,1,1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", 
                torch.ones(self.K-1,1,1,1,1)*0.1, constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", 
                torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        
        # AoI Local Parameters
        b_v = pyro.param("b_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        h_alpha = pyro.param("h_alpha", self.data.height.mean()*torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        w_v = pyro.param("w_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,self.F,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,self.F,1,1), constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,self.F,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,self.F,1,1), constraint=constraints.positive)
        
        # AoI & Frame Local Parameters
        #z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)
        
        # Global Variables
        pyro.sample("pi", dist.Dirichlet(pi_concentration))
        pyro.sample("background_loc", dist.Delta(background_loc_v))
        pyro.sample("background_scale", dist.Delta(background_scale_v))
        with K_plate:
            pyro.sample("height_loc", dist.Delta(height_loc_v))
            pyro.sample("height_scale", dist.Delta(height_scale_v))
            pyro.sample("width_alpha", dist.Delta(width_alpha_v))
            pyro.sample("width_beta", dist.Delta(width_beta_v))
            pyro.sample("x0_scale", dist.Delta(x0_scale_v))
            pyro.sample("y0_scale", dist.Delta(y0_scale_v))
            with N_plate:
                pyro.sample("height", dist.Gamma(h_alpha[:,batch_idx], h_beta[:,batch_idx]))
                pyro.sample("width", dist.Delta(w_v[:,batch_idx]))
                with F_plate: 
                    pyro.sample("x0", dist.Normal(x_loc[:,batch_idx][:,:,frame_idx], x_scale[:,batch_idx][:,:,frame_idx]))
                    pyro.sample("y0", dist.Normal(y_loc[:,batch_idx][:,:,frame_idx], y_scale[:,batch_idx][:,:,frame_idx]))
                
        
        with N_plate:
            # AoI Local Variables
            pyro.sample("background", dist.Delta(b_v[:,batch_idx]))
            
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                #background = pyro.sample("background", dist.Delta(background_v[:,batch_idx][:,:,frame_idx]))
 

