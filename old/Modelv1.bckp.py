import numpy as np
import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
from pyro.contrib.autoguide import AutoDelta
import pyro
from pyro import poutine
from pyro.infer import  config_enumerate
import pyro.distributions as dist
import pickle


class Modelv1:
    """ Gaussian Spot Model """
    def __init__(self, data=None, K=None):
        
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.data = data
        self.N, self.F, self.D, _ = data._store.shape
        self.K = K
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        #self.pixel_pos = torch.tensor(np.indices((self.D,self.D))).float()
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).float()
        self.pixel_pos = self.pixel_pos.reshape(1,1,1,self.D,self.D,2)
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.F,2) + self.data.target[["x", "y"]].values.reshape(self.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(1,self.N,self.F,1,1,2).repeat(self.K-1,1,1,1,1,1)
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,1,2,2)
        
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
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * np.pi * width**2
        # height can be either a scalar or a vector
        # 1,K,ind,F,D,D
        return height * gaussian_spot #
    
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        #plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))
        
        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
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
                x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                y0 = pyro.sample("y0", dist.Normal(0., y0_scale))
                
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                #background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]
    
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        # plates
        K_plate = pyro.plate("K_plate", self.K-1, dim=-5)
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", self.F, subsample=frame_idx, dim=-3)
        
        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", 
                torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", 
                self.data.background.mean(), constraint=constraints.positive)
        background_scale_v = pyro.param("background_scale_v", 
                torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", 
                torch.ones(self.K-1,1,1,1,1)*100, constraint=constraints.positive)
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
        b_v = pyro.param("background_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        h_v = pyro.param("h_v", 100*torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        w_v = pyro.param("w_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_v = pyro.param("x_v", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y_v = pyro.param("y_v", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        
        # AoI & Frame Local Parameters
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        #z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)
        
        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
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
                height = pyro.sample("height", dist.Delta(h_v[:,batch_idx]))
                width = pyro.sample("width", dist.Delta(w_v[:,batch_idx]))
                x0 = pyro.sample("x0", dist.Delta(x_v[:,batch_idx]))
                y0 = pyro.sample("y0", dist.Delta(y_v[:,batch_idx]))
                
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Delta(b_v[:,batch_idx]))
            
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                #background = pyro.sample("background", dist.Delta(background_v[:,batch_idx][:,:,frame_idx]))
        
        #return height, width, background, x0, y0

    
    def locs_mcmc(self):
        N_plate = pyro.plate("N_axis", self.N, dim=-4)
        F_plate = pyro.plate("F_axis", size=self.F, dim=-3)
        
        # global locs variables
        width = pyro.sample("width", dist.Gamma(1, 0.1).expand([1,1,1,1,1]))
        with N_plate:
            #background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
            with F_plate:
                #z = pyro.sample("z", dist.Categorical(weights))
                # local height and locs
                #width = pyro.sample("width", dist.Gamma(1, 0.1).expand([1,1,1,1,1]))
                background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
                height = pyro.sample("height", dist.HalfNormal(500.).expand([1,1,1,1,1]))
                x0 = pyro.sample("x0", dist.Normal(0.,10.).expand([1,1,1,1,1]))
                y0 = pyro.sample("y0", dist.Normal(0.,10.).expand([1,1,1,1,1]))
        #locs = torch.ones(1, len(batch_idx), len(frame_idx), self.D, self.D) * background
        #locs = self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0) + background
        locs = self.gaussian_spot(torch.arange(self.N), torch.arange(self.F), height, width, x0, y0) + background
        #print(background.shape, x0.shape, locs.shape)
        return locs[0]

    def locs_feature_model(self, batch_idx, frame_idx):
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", size=self.F, subsample=frame_idx, dim=-3)
        
        # global locs variables
        width = pyro.sample("width", dist.Gamma(1, 0.1).expand([1,1,1,1,1]))
        with N_plate:
            #background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
            with F_plate:
                #z = pyro.sample("z", dist.Categorical(weights))
                # local height and locs
                #width = pyro.sample("width", dist.Gamma(1, 0.1).expand([1,1,1,1,1]))
                background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
                height = pyro.sample("height", dist.HalfNormal(500.).expand([1,1,1,1,1]))
                x0 = pyro.sample("x0", dist.Normal(0.,10.).expand([1,1,1,1,1]))
                y0 = pyro.sample("y0", dist.Normal(0.,10.).expand([1,1,1,1,1]))
        #locs = torch.ones(1, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs = self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0) + background
        #locs = self.gaussian_spot(torch.arange(len(self.data)), height, width, x0, y0) + background
        #print(background.shape, x0.shape, locs.shape)
        return locs[0]


    def locs_feature_guide(self, batch_idx, frame_idx):
        N_plate = pyro.plate("N_plate", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("F_plate", size=self.F, subsample=frame_idx, dim=-3)
        
        # global locs variables
        background_loc = pyro.param("background_loc", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        background_beta = pyro.param("background_beta", torch.ones(1,self.N,self.F,1,1), constraint=constraints.positive)
        
        width_loc = pyro.param("width_loc", torch.ones(1,1,1,1,1)*1.5, constraint=constraints.interval(0,5))
        width_beta = pyro.param("width_beta", torch.ones(1,1,1,1,1), constraint=constraints.positive)
        
        # local variables
        x0_loc = pyro.param("x0_loc", torch.zeros(1,self.N,self.F,1,1), constraint=constraints.real)
        x0_scale = pyro.param("x0_scale", torch.ones(1,self.N,self.F,1,1), constraint=constraints.positive)
        y0_loc = pyro.param("y0_loc", torch.zeros(1,self.N,self.F,1,1), constraint=constraints.real)
        y0_scale = pyro.param("y0_scale", torch.ones(1,self.N,self.F,1,1), constraint=constraints.positive)
        height_loc = pyro.param("height_loc", self.data.height.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        height_beta = pyro.param("height_beta", torch.ones(1,self.N,self.F,1,1), constraint=constraints.positive)
        
        pyro.sample("width", dist.Gamma(width_loc*width_beta, width_beta))
        with N_plate:
            #pyro.sample("background", dist.Delta(background_delta[:,batch_idx]))
            with F_plate:
                # local height and locs
                pyro.sample("background", dist.Gamma(background_loc[:,batch_idx][:,:,frame_idx]*background_beta[:,batch_idx][:,:,frame_idx], background_beta[:,batch_idx][:,:,frame_idx]))
                pyro.sample("height", dist.Gamma(height_loc[:,batch_idx][:,:,frame_idx]*height_beta[:,batch_idx][:,:,frame_idx], height_beta[:,batch_idx][:,:,frame_idx]))
                pyro.sample("x0", dist.Normal(x0_loc[:,batch_idx][:,:,frame_idx], x0_scale[:,batch_idx][:,:,frame_idx]))
                pyro.sample("y0", dist.Normal(y0_loc[:,batch_idx][:,:,frame_idx], y0_scale[:,batch_idx][:,:,frame_idx]))
        
    def locs_sample(self):
        background = pyro.param("background_loc")
        height = pyro.param("height_loc")
        width = pyro.param("width_delta")
        x0 = pyro.param("x_delta")
        y0 = pyro.param("y_delta")
        # return locs for K classes
        locs = torch.ones(self.K, len(self.data), self.D, self.D) * background
        for k in range(1,self.K):
            locs[k,:,:,:] += self.gaussian_spot(torch.arange(len(self.data)), height[k-1], width[k-1], x0[k-1], y0[k-1])
        return locs


class Modelv1p4(Modelv1):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        """
        Global Variables:
            weights {Dirichlet(0.5)}
            background_loc {HalfNormal(1000)}
            background_scale {HalfNormal(100)}
            height_loc {HalfNormal(500)}
            height_scale {HalfNormal(50)}
            width_alpha {HalfNormal(50)}
            width_beta {HalfNormal(5)}
        
        AoI Local Variables:
            background {Normal(background_loc, background_scale)}
            height {Normal(height_loc, height_scale)}
            width {Gamma(width_alpha, widht_beta)}
            x0 {Normal(0, 1)}
            y0 {Normal(0, 1)}
            
        AoI & Frame Local Variables:
            z {Categorical(weights)}
        """
        #plates
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))
        
        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_scale = pyro.sample("background_scale", dist.HalfNormal(100.).expand([1,1,1,1,1]))
        height_loc = pyro.sample("height_loc", dist.HalfNormal(500.).expand([self.K-1,1,1,1,1]))
        height_scale = pyro.sample("height_scale", dist.HalfNormal(50.).expand([self.K-1,1,1,1,1]))
        width_alpha = pyro.sample("width_alpha", dist.HalfNormal(50.).expand([self.K-1,1,1,1,1]))
        width_beta = pyro.sample("width_beta", dist.HalfNormal(5.).expand([self.K-1,1,1,1,1]))
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            height = pyro.sample("height", dist.Normal(height_loc, height_scale))
            width = pyro.sample("width", dist.Gamma(width_alpha, width_beta))
            x0 = pyro.sample("x0", dist.Normal(0., 1.))
            y0 = pyro.sample("y0", dist.Normal(0., 1.))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                #background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]
    
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        """
        Global Variables & Guides:
            weights {Dirichlet(weights_concentration=N*F/K)}
            background_loc {Delta(background_loc_v=background.mean())}
            background_scale {Delta(background_scale_v=10)}
            height_loc {Delta(height_loc_v=100)}
            height_scale {Delta(height_scale_v=10)}
            width_alpha {Delta(width_alpha_v=0.15)}
            width_beta {Delta(width_beta_v=0.1)}
        
        AoI Local Variables:
            background {Delta(background_v=background.mean(dim=1))}
            height {Delta(height_v=100)}
            width {Delta(width_v=1.5)}
            x0 {Delta(x0_v=0)} 
            y0 {Delta(y0_v=0)}
            
        AoI & Frame Local Variables:
            z {Categorical(z_probs=0.5)}
        """
        # plates
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        
        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.background.mean(), constraint=constraints.positive)
        background_scale_v = pyro.param("background_scale_v", torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", torch.ones(self.K-1)*100, constraint=constraints.positive)
        height_scale_v = pyro.param("height_scale_v", torch.ones(self.K-1)*10, constraint=constraints.positive)
        width_alpha_v = pyro.param("width_alpha_v", torch.ones(self.K-1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1)*0.1, constraint=constraints.positive)
        
        # AoI Local Parameters
        background_v = pyro.param("background_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        #background_delta = pyro.param("background_delta", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        height_v = pyro.param("height_v", torch.ones(self.K-1,self.N,1,1,1)*100, constraint=constraints.positive)
        width_v = pyro.param("width_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x0_v = pyro.param("x0_v", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y0_v = pyro.param("y0_v", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        
        # AoI & Frame Local Parameters
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
        pyro.sample("background_loc", dist.Delta(background_loc_v))
        pyro.sample("background_scale", dist.Delta(background_scale_v))
        pyro.sample("height_loc", dist.Delta(height_loc_v).expand([self.K-1,1,1,1,1]))
        pyro.sample("height_scale", dist.Delta(height_scale_v).expand([self.K-1,1,1,1,1]))
        pyro.sample("width_alpha", dist.Delta(width_alpha_v).expand([self.K-1,1,1,1,1]))
        pyro.sample("width_beta", dist.Delta(width_beta_v).expand([self.K-1,1,1,1,1]))
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Delta(background_v[:,batch_idx]))
            height = pyro.sample("height", dist.Delta(height_v[:,batch_idx]))
            width = pyro.sample("width", dist.Delta(width_v[:,batch_idx]))
            x0 = pyro.sample("x0", dist.Delta(x0_v[:,batch_idx]))
            y0 = pyro.sample("y0", dist.Delta(y0_v[:,batch_idx]))
            
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                #background = pyro.sample("background", dist.Delta(background_delta[:,batch_idx][:,:,frame_idx]))
        
        return height, width, background, x0, y0
       
class Modelv1p5(Modelv1):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        """
        Global Variables:
            weights {Dirichlet(0.5)}
            background_loc {HalfNormal(1000)}
            background_scale {HalfNormal(100)}
            height_loc {HalfNormal(500)}
            height_scale {HalfNormal(50)}
            width_alpha {HalfNormal(50)}
            width_beta {HalfNormal(5)}
        
        AoI Local Variables:
            background {Normal(background_loc, background_scale)}
            height {Normal(height_loc, height_scale)}
            width {Gamma(width_alpha, widht_beta)}
            x0 {Normal(0, 1)}
            y0 {Normal(0, 1)}
            
        AoI & Frame Local Variables:
            z {Categorical(weights)}
        """
        #plates
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))
        
        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_scale = pyro.sample("background_scale", dist.HalfNormal(100.).expand([1,1,1,1,1]))
        height_loc = pyro.sample("height_loc", dist.HalfNormal(500.).expand([self.K-1,1,1,1,1]))
        height_scale = pyro.sample("height_scale", dist.HalfNormal(50.).expand([self.K-1,1,1,1,1]))
        width_alpha = pyro.sample("width_alpha", dist.HalfNormal(50.).expand([self.K-1,1,1,1,1]))
        width_beta = pyro.sample("width_beta", dist.HalfNormal(5.).expand([self.K-1,1,1,1,1]))
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            height = pyro.sample("height", dist.Normal(height_loc, height_scale))
            width = pyro.sample("width", dist.Gamma(width_alpha, width_beta))
            x0 = pyro.sample("x0", dist.Normal(0., 1.))
            y0 = pyro.sample("y0", dist.Normal(0., 1.))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                #background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]
    
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        """
        Global Variables & Guides:
            weights {Dirichlet(weights_concentration=N*F/K)}
            background_loc {Delta(background_loc_v=background.mean())}
            background_scale {Delta(background_scale_v=10)}
            height_loc {Delta(height_loc_v=100)}
            height_scale {Delta(height_scale_v=10)}
            width_alpha {Delta(width_alpha_v=0.15)}
            width_beta {Delta(width_beta_v=0.1)}
        
        AoI Local Variables:
            background {Delta(background_v=background.mean(dim=1))}
            height {Delta(height_v=100)}
            width {Delta(width_v=1.5)}
            x0 {Delta(x0_v=0)} 
            y0 {Delta(y0_v=0)}
            
        AoI & Frame Local Variables:
            z {Categorical(z_probs=0.5)}
        """
        # plates
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        
        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.background.mean(), constraint=constraints.positive)
        background_scale_v = pyro.param("background_scale_v", torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", torch.ones(self.K-1)*100, constraint=constraints.positive)
        height_scale_v = pyro.param("height_scale_v", torch.ones(self.K-1)*10, constraint=constraints.positive)
        width_alpha_v = pyro.param("width_alpha_v", torch.ones(self.K-1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1)*0.1, constraint=constraints.positive)
        
        # AoI Local Parameters
        background_v = pyro.param("background_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        #background_delta = pyro.param("background_delta", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        height_v = pyro.param("height_v", torch.ones(self.K-1,self.N,1,1,1)*100, constraint=constraints.positive)
        width_v = pyro.param("width_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x0_v = pyro.param("x0_v", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y0_v = pyro.param("y0_v", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        
        # AoI & Frame Local Parameters
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        
        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
        pyro.sample("background_loc", dist.Delta(background_loc_v))
        pyro.sample("background_scale", dist.Delta(background_scale_v))
        pyro.sample("height_loc", dist.Delta(height_loc_v).expand([self.K-1,1,1,1,1]))
        pyro.sample("height_scale", dist.Delta(height_scale_v).expand([self.K-1,1,1,1,1]))
        pyro.sample("width_alpha", dist.Delta(width_alpha_v).expand([self.K-1,1,1,1,1]))
        pyro.sample("width_beta", dist.Delta(width_beta_v).expand([self.K-1,1,1,1,1]))
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Delta(background_v[:,batch_idx]))
            height = pyro.sample("height", dist.Delta(height_v[:,batch_idx]))
            width = pyro.sample("width", dist.Delta(width_v[:,batch_idx]))
            x0 = pyro.sample("x0", dist.Delta(x0_v[:,batch_idx]))
            y0 = pyro.sample("y0", dist.Delta(y0_v[:,batch_idx]))
            
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                #background = pyro.sample("background", dist.Delta(background_delta[:,batch_idx][:,:,frame_idx]))
        
        return height, width, background, x0, y0

    
    
class Modelv1p6(Modelv1p4):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        """
        Global Variables:
            weights {Dirichlet(0.5)}
            background_loc {HalfNormal(1000)}
            background_scale {HalfNormal(100)}
            height_loc {HalfNormal(500)}
            height_scale {HalfNormal(50)}
            width_alpha {HalfNormal(50)}
            width_beta {HalfNormal(5)}
            x0_scale {HalfNormal(5)}
            y0_scale {HalfNormal(5)}
        
        AoI Local Variables:
            background {Normal(background_loc, background_scale)}
            height {Normal(height_loc, height_scale)}
            width {Gamma(width_alpha, widht_beta)}
            x0 {Normal(0, x0_scale)}
            y0 {Normal(0, y0_scale)}
            
        AoI & Frame Local Variables:
            z {Categorical(weights)}
        """
        #plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))
        
        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1,1,1]))
        
        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(5.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(5.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(5.))
            with N_plate:
                height = pyro.sample("height", dist.Gamma(height_loc, height_beta))
                width = pyro.sample("width", dist.Gamma(width_loc, width_beta))
                x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                y0 = pyro.sample("y0", dist.Normal(0., y0_scale))
                
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Gamma(background_loc, background_beta))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                #background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]
    
    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        """
        Global Variables & Guides:
            weights {Dirichlet(weights_concentration=N*F/K)}
            background_loc {Delta(background_loc_v=background.mean())}
            background_scale {Delta(background_scale_v=10)}
            height_loc {Delta(height_loc_v=100)}
            height_scale {Delta(height_scale_v=10)}
            width_alpha {Delta(width_alpha_v=0.15)}
            width_beta {Delta(width_beta_v=0.1)}
            x0_scale {Delta(x0_scale_v=0.5)}
            y0_scale {Delta(y0_scale_v=0.5)}
        
        AoI Local Variables:
            background {Delta(background_v=background.mean(dim=1))}
            height {Delta(height_v=100)}
            width {Delta(width_v=1.5)}
            x0 {Delta(x0_v=0)} 
            y0 {Delta(y0_v=0)}
            
        AoI & Frame Local Variables:
            z {Categorical(z_probs=0.5)}
        """
        # plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        
        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.background.mean(), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", torch.ones(self.K-1,1,1,1,1)*100, constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", torch.ones(self.K-1,1,1,1,1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1,1,1,1,1)*0.1, constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        
        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        b_beta = pyro.param("b_beta", torch.ones(1,self.N,1,1,1), constraint=constraints.positive)
        h_loc = pyro.param("h_loc", torch.ones(self.K-1,self.N,1,1,1)*100, constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        width_v = pyro.param("width_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,1,1,1)*0.1, constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,1,1,1)*0.1, constraint=constraints.positive)
        
        # AoI & Frame Local Parameters
        #z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)
        
        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
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
                height = pyro.sample("height", dist.Gamma(h_loc[:,batch_idx], h_beta[:,batch_idx]))
                width = pyro.sample("width", dist.Delta(width_v[:,batch_idx]))
                x0 = pyro.sample("x0", dist.Normal(x_loc[:,batch_idx], x_scale[:,batch_idx]))
                y0 = pyro.sample("y0", dist.Normal(y_loc[:,batch_idx], y_scale[:,batch_idx]))
                
        
        with N_plate:
            # AoI Local Variables
            background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx], b_beta[:,batch_idx]))
            #background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx]))
            
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                #background = pyro.sample("background", dist.Delta(background_v[:,batch_idx][:,:,frame_idx]))
        
        return height, width, background, x0, y0

# 1.8
class Modelv1p8(Modelv1p4):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        """
        Global Variables:
            weights {Dirichlet(0.5)}
            background_loc {HalfNormal(1000)}
            background_scale {HalfNormal(100)}
            height_loc {HalfNormal(500)}
            height_scale {HalfNormal(50)}
            width_alpha {HalfNormal(50)}
            width_beta {HalfNormal(5)}
            x0_scale {HalfNormal(5)}
            y0_scale {HalfNormal(5)}
        
       AoI Local Variables:
            background {Normal(background_loc, background_scale)}
            height {Normal(height_loc, height_scale)}
            width {Gamma(width_alpha, widht_beta)}
            x0 {Normal(0, x0_scale)}
            y0 {Normal(0, y0_scale)}

        AoI & Frame Local Variables:
            z {Categorical(weights)}
        """
        #plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))

        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        

        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(5.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(5.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(5.))
            with N_plate:
                width = pyro.sample("width", dist.Gamma(width_loc, width_beta))
                height = pyro.sample("height", dist.Gamma(height_loc, height_beta))
                x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                y0 = pyro.sample("y0", dist.Normal(0., y0_scale))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(background_loc, background_beta))
            background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.))
            background_beta = pyro.sample("background_beta", dist.HalfNormal(100.))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                background = pyro.sample("background", dist.Gamma(background_loc, background_beta))

        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]

    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        """
        Global Variables & Guides:
            weights {Dirichlet(weights_concentration=N*F/K)}
            background_loc {Delta(background_loc_v=background.mean())}
            background_scale {Delta(background_scale_v=10)}
            height_loc {Delta(height_loc_v=100)}
            height_scale {Delta(height_scale_v=10)}
            width_alpha {Delta(width_alpha_v=0.15)}
            width_beta {Delta(width_beta_v=0.1)}
            x0_scale {Delta(x0_scale_v=0.5)}
            y0_scale {Delta(y0_scale_v=0.5)}

        AoI Local Variables:
            background {Delta(background_v=background.mean(dim=1))}
            height {Delta(height_v=100)}
            width {Delta(width_v=1.5)}
            x0 {Delta(x0_v=0)} 
            y0 {Delta(y0_v=0)}

        AoI & Frame Local Variables:
            z {Categorical(z_probs=0.5)}
        """
        # plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)

        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", torch.ones(1,self.N,1,1,1), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", torch.ones(self.K-1,1,1,1,1)*100, constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", torch.ones(self.K-1,1,1,1,1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1,1,1,1,1)*0.1, constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)

        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        #b_beta = pyro.param("b_beta", torch.ones(1,self.N,self.F,1,1)*10, constraint=constraints.positive)
        h_loc = pyro.param("h_loc", torch.ones(self.K-1,self.N,1,1,1)*100, constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        width_v = pyro.param("width_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,1,1,1)*0.1, constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,1,1,1)*0.1, constraint=constraints.positive)

        # AoI & Frame Local Parameters
        #z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)

        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
        
        with K_plate:
            pyro.sample("height_loc", dist.Delta(height_loc_v))
            pyro.sample("height_beta", dist.Delta(height_beta_v))
            pyro.sample("width_loc", dist.Delta(width_loc_v))
            pyro.sample("width_beta", dist.Delta(width_beta_v))
            pyro.sample("x0_scale", dist.Delta(x0_scale_v))
            pyro.sample("y0_scale", dist.Delta(y0_scale_v))
            with N_plate:
                width = pyro.sample("width", dist.Delta(width_v[:,batch_idx]))
                height = pyro.sample("height", dist.Gamma(h_loc[:,batch_idx], h_beta[:,batch_idx]))
                x0 = pyro.sample("x0", dist.Normal(x_loc[:,batch_idx], x_scale[:,batch_idx]))
                y0 = pyro.sample("y0", dist.Normal(y_loc[:,batch_idx], y_scale[:,batch_idx]))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx], b_beta[:,batch_idx]))
            #background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx]))
            pyro.sample("background_loc", dist.Delta(background_loc_v[:,batch_idx]))
            pyro.sample("background_beta", dist.Delta(background_beta_v[:,batch_idx]))
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx][:,:,frame_idx]))
                #background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx][:,:,frame_idx], b_beta[:,batch_idx][:,:,frame_idx]))

        return height, width, background, x0, y0
    
# 1.8
class Modelv1p7(Modelv1p4):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        """
        Global Variables:
            weights {Dirichlet(0.5)}
            background_loc {HalfNormal(1000)}
            background_scale {HalfNormal(100)}
            height_loc {HalfNormal(500)}
            height_scale {HalfNormal(50)}
            width_alpha {HalfNormal(50)}
            width_beta {HalfNormal(5)}
            x0_scale {HalfNormal(5)}
            y0_scale {HalfNormal(5)}
        
       AoI Local Variables:
            background {Normal(background_loc, background_scale)}
            height {Normal(height_loc, height_scale)}
            width {Gamma(width_alpha, widht_beta)}
            x0 {Normal(0, x0_scale)}
            y0 {Normal(0, y0_scale)}

        AoI & Frame Local Variables:
            z {Categorical(weights)}
        """
        #plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))

        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_beta = pyro.sample("background_beta", dist.HalfNormal(100.).expand([1,1,1,1,1]))

        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(5.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(5.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(5.))
            with N_plate:
                width = pyro.sample("width", dist.Gamma(width_loc, width_beta))
                height = pyro.sample("height", dist.Gamma(height_loc, height_beta))
                x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                y0 = pyro.sample("y0", dist.Normal(0., y0_scale))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(background_loc, background_beta))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                background = pyro.sample("background", dist.Gamma(background_loc, background_beta))

        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]

    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        """
        Global Variables & Guides:
            weights {Dirichlet(weights_concentration=N*F/K)}
            background_loc {Delta(background_loc_v=background.mean())}
            background_scale {Delta(background_scale_v=10)}
            height_loc {Delta(height_loc_v=100)}
            height_scale {Delta(height_scale_v=10)}
            width_alpha {Delta(width_alpha_v=0.15)}
            width_beta {Delta(width_beta_v=0.1)}
            x0_scale {Delta(x0_scale_v=0.5)}
            y0_scale {Delta(y0_scale_v=0.5)}

        AoI Local Variables:
            background {Delta(background_v=background.mean(dim=1))}
            height {Delta(height_v=100)}
            width {Delta(width_v=1.5)}
            x0 {Delta(x0_v=0)} 
            y0 {Delta(y0_v=0)}

        AoI & Frame Local Variables:
            z {Categorical(z_probs=0.5)}
        """
        # plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)

        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.background.mean(), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", torch.tensor(10.), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", torch.ones(self.K-1,1,1,1,1)*100, constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", torch.ones(self.K-1,1,1,1,1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1,1,1,1,1)*0.1, constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)

        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.background.reshape(1,self.N,self.F,1,1)*10, constraint=constraints.positive)
        b_beta = pyro.param("b_beta", torch.ones(1,self.N,self.F,1,1)*10, constraint=constraints.positive)
        h_loc = pyro.param("h_loc", torch.ones(self.K-1,self.N,1,1,1)*100, constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.K-1,self.N,1,1,1), constraint=constraints.positive)
        width_v = pyro.param("width_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,1,1,1)*0.1, constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,1,1,1)*0.1, constraint=constraints.positive)

        # AoI & Frame Local Parameters
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)

        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
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
                width = pyro.sample("width", dist.Delta(width_v[:,batch_idx]))
                height = pyro.sample("height", dist.Gamma(h_loc[:,batch_idx], h_beta[:,batch_idx]))
                x0 = pyro.sample("x0", dist.Normal(x_loc[:,batch_idx], x_scale[:,batch_idx]))
                y0 = pyro.sample("y0", dist.Normal(y_loc[:,batch_idx], y_scale[:,batch_idx]))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx], b_beta[:,batch_idx]))
            #background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx]))

            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx][:,:,frame_idx], b_beta[:,batch_idx][:,:,frame_idx]))

        return height, width, background, x0, y0
    
    
class Modelv1p9(Modelv1p4):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        """
        Global Variables:
            weights {Dirichlet(0.5)}
            background_loc {HalfNormal(1000)}
            background_scale {HalfNormal(100)}
            height_loc {HalfNormal(500)}
            height_scale {HalfNormal(50)}
            width_alpha {HalfNormal(50)}
            width_beta {HalfNormal(5)}
            x0_scale {HalfNormal(5)}
            y0_scale {HalfNormal(5)}
        
       AoI Local Variables:
            background {Normal(background_loc, background_scale)}
            height {Normal(height_loc, height_scale)}
            width {Gamma(width_alpha, widht_beta)}
            x0 {Normal(0, x0_scale)}
            y0 {Normal(0, y0_scale)}

        AoI & Frame Local Variables:
            z {Categorical(weights)}
        """
        #plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))

        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        

        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(5.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(5.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(5.))
            with N_plate:
                width = pyro.sample("width", dist.Gamma(width_loc, width_beta))
                with F_plate:
                    height = pyro.sample("height", dist.Gamma(height_loc, height_beta))
                    x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                    y0 = pyro.sample("y0", dist.Normal(0., y0_scale))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(background_loc, background_beta))
            background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.))
            background_beta = pyro.sample("background_beta", dist.HalfNormal(100.))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                background = pyro.sample("background", dist.Gamma(background_loc, background_beta))

        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]

    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        """
        Global Variables & Guides:
            weights {Dirichlet(weights_concentration=N*F/K)}
            background_loc {Delta(background_loc_v=background.mean())}
            background_scale {Delta(background_scale_v=10)}
            height_loc {Delta(height_loc_v=100)}
            height_scale {Delta(height_scale_v=10)}
            width_alpha {Delta(width_alpha_v=0.15)}
            width_beta {Delta(width_beta_v=0.1)}
            x0_scale {Delta(x0_scale_v=0.5)}
            y0_scale {Delta(y0_scale_v=0.5)}

        AoI Local Variables:
            background {Delta(background_v=background.mean(dim=1))}
            height {Delta(height_v=100)}
            width {Delta(width_v=1.5)}
            x0 {Delta(x0_v=0)} 
            y0 {Delta(y0_v=0)}

        AoI & Frame Local Variables:
            z {Categorical(z_probs=0.5)}
        """
        # plates
        K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)

        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", torch.ones(1,self.N,1,1,1), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", torch.ones(self.K-1,1,1,1,1)*100, constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K-1,1,1,1,1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", torch.ones(self.K-1,1,1,1,1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K-1,1,1,1,1)*0.1, constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", torch.ones(self.K-1,1,1,1,1)*0.5, constraint=constraints.positive)

        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        #b_beta = pyro.param("b_beta", torch.ones(1,self.N,self.F,1,1)*10, constraint=constraints.positive)
        h_loc = pyro.param("h_loc", torch.ones(self.K-1,self.N,self.F,1,1)*100, constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.K-1,self.N,self.F,1,1), constraint=constraints.positive)
        width_v = pyro.param("width_v", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K-1,self.N,self.F,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K-1,self.N,self.F,1,1)*0.1, constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K-1,self.N,self.F,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K-1,self.N,self.F,1,1)*0.1, constraint=constraints.positive)

        # AoI & Frame Local Parameters
        #z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)

        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
        
        with K_plate:
            pyro.sample("height_loc", dist.Delta(height_loc_v))
            pyro.sample("height_beta", dist.Delta(height_beta_v))
            pyro.sample("width_loc", dist.Delta(width_loc_v))
            pyro.sample("width_beta", dist.Delta(width_beta_v))
            pyro.sample("x0_scale", dist.Delta(x0_scale_v))
            pyro.sample("y0_scale", dist.Delta(y0_scale_v))
            with N_plate:
                width = pyro.sample("width", dist.Delta(width_v[:,batch_idx]))
                with F_plate:
                    height = pyro.sample("height", dist.Gamma(h_loc[:,batch_idx][:,:,frame_idx], h_beta[:,batch_idx][:,:,frame_idx]))
                    x0 = pyro.sample("x0", dist.Normal(x_loc[:,batch_idx][:,:,frame_idx], x_scale[:,batch_idx][:,:,frame_idx]))
                    y0 = pyro.sample("y0", dist.Normal(y_loc[:,batch_idx][:,:,frame_idx], y_scale[:,batch_idx][:,:,frame_idx]))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx], b_beta[:,batch_idx]))
            #background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx]))
            pyro.sample("background_loc", dist.Delta(background_loc_v[:,batch_idx]))
            pyro.sample("background_beta", dist.Delta(background_beta_v[:,batch_idx]))
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx][:,:,frame_idx]))
                #background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx][:,:,frame_idx], b_beta[:,batch_idx][:,:,frame_idx]))

        return height, width, background, x0, y0
    
    
class Modelv1p10(Modelv1p4):
    @config_enumerate
    def locs_mixture_model(self, batch_idx, frame_idx):
        """
        Global Variables:
            weights {Dirichlet(0.5)}
            background_loc {HalfNormal(1000)}
            background_scale {HalfNormal(100)}
            height_loc {HalfNormal(500)}
            height_scale {HalfNormal(50)}
            width_alpha {HalfNormal(50)}
            width_beta {HalfNormal(5)}
            x0_scale {HalfNormal(5)}
            y0_scale {HalfNormal(5)}
        
       AoI Local Variables:
            background {Normal(background_loc, background_scale)}
            height {Normal(height_loc, height_scale)}
            width {Gamma(width_alpha, widht_beta)}
            x0 {Normal(0, x0_scale)}
            y0 {Normal(0, y0_scale)}

        AoI & Frame Local Variables:
            z {Categorical(weights)}
        """
        #plates
        K_plate = pyro.plate("component_axis", self.K, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))

        # Global Variables
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        

        with K_plate:
            height_loc = pyro.sample("height_loc", dist.HalfNormal(500.))
            height_beta = pyro.sample("height_beta", dist.HalfNormal(50.))
            width_loc = pyro.sample("width_loc", dist.HalfNormal(5.))
            width_beta = pyro.sample("width_beta", dist.HalfNormal(5.))
            x0_scale = pyro.sample("x0_scale", dist.HalfNormal(5.))
            y0_scale = pyro.sample("y0_scale", dist.HalfNormal(5.))
            with N_plate:
                width = pyro.sample("width", dist.Gamma(width_loc, width_beta))
                height = pyro.sample("height", dist.Gamma(height_loc, height_beta))
                x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
                y0 = pyro.sample("y0", dist.Normal(0., y0_scale))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(background_loc, background_beta))
            background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.))
            background_beta = pyro.sample("background_beta", dist.HalfNormal(100.))
            with F_plate:
                # AoI & Frame Local Variables
                z = pyro.sample("z", dist.Categorical(weights))
                background = pyro.sample("background", dist.Gamma(background_loc, background_beta))

        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        #locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)
        locs += self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)

        return locs[z,nind,find,xind,yind]

    @config_enumerate
    def locs_mixture_guide(self, batch_idx, frame_idx):
        """
        Global Variables & Guides:
            weights {Dirichlet(weights_concentration=N*F/K)}
            background_loc {Delta(background_loc_v=background.mean())}
            background_scale {Delta(background_scale_v=10)}
            height_loc {Delta(height_loc_v=100)}
            height_scale {Delta(height_scale_v=10)}
            width_alpha {Delta(width_alpha_v=0.15)}
            width_beta {Delta(width_beta_v=0.1)}
            x0_scale {Delta(x0_scale_v=0.5)}
            y0_scale {Delta(y0_scale_v=0.5)}

        AoI Local Variables:
            background {Delta(background_v=background.mean(dim=1))}
            height {Delta(height_v=100)}
            width {Delta(width_v=1.5)}
            x0 {Delta(x0_v=0)} 
            y0 {Delta(y0_v=0)}

        AoI & Frame Local Variables:
            z {Categorical(z_probs=0.5)}
        """
        # plates
        K_plate = pyro.plate("component_axis", self.K, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)

        # Global Parameters
        weights_concentration = pyro.param("weights_concentration", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        background_loc_v = pyro.param("background_loc_v", self.data.background.mean(dim=1).reshape(1,self.N,1,1,1), constraint=constraints.positive)
        background_beta_v = pyro.param("background_beta_v", torch.ones(1,self.N,1,1,1), constraint=constraints.positive)
        height_loc_v = pyro.param("height_loc_v", torch.ones(self.K,1,1,1,1)*100, constraint=constraints.positive)
        height_beta_v = pyro.param("height_beta_v", torch.ones(self.K,1,1,1,1)*10, constraint=constraints.positive)
        width_loc_v = pyro.param("width_loc_v", torch.ones(self.K,1,1,1,1)*0.15, constraint=constraints.positive)
        width_beta_v = pyro.param("width_beta_v", torch.ones(self.K,1,1,1,1)*0.1, constraint=constraints.positive)
        x0_scale_v = pyro.param("x0_scale_v", torch.ones(self.K,1,1,1,1)*0.5, constraint=constraints.positive)
        y0_scale_v = pyro.param("y0_scale_v", torch.ones(self.K,1,1,1,1)*0.5, constraint=constraints.positive)

        # AoI Local Parameters
        b_loc = pyro.param("b_loc", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        #b_beta = pyro.param("b_beta", torch.ones(1,self.N,self.F,1,1)*10, constraint=constraints.positive)
        h_loc = pyro.param("h_loc", torch.ones(self.K,self.N,1,1,1)*100, constraint=constraints.positive)
        h_beta = pyro.param("h_beta", torch.ones(self.K,self.N,1,1,1), constraint=constraints.positive)
        width_v = pyro.param("width_v", torch.ones(self.K,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x_loc = pyro.param("x_loc", torch.zeros(self.K,self.N,1,1,1), constraint=constraints.real)
        x_scale = pyro.param("x_scale", torch.ones(self.K,self.N,1,1,1)*0.1, constraint=constraints.positive)
        y_loc = pyro.param("y_loc", torch.zeros(self.K,self.N,1,1,1), constraint=constraints.real)
        y_scale = pyro.param("y_scale", torch.ones(self.K,self.N,1,1,1)*0.1, constraint=constraints.positive)

        # AoI & Frame Local Parameters
        #z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)
        z_probs = pyro.param("z_probs", self.data.probs.reshape(self.N,self.F,1,1,self.K), constraint=constraints.simplex)
        #background_v = pyro.param("background_v", self.data.background.reshape(1,self.N,1,1,1), constraint=constraints.positive)

        # Global Variables
        pyro.sample("weights", dist.Dirichlet(weights_concentration))
        
        with K_plate:
            pyro.sample("height_loc", dist.Delta(height_loc_v))
            pyro.sample("height_beta", dist.Delta(height_beta_v))
            pyro.sample("width_loc", dist.Delta(width_loc_v))
            pyro.sample("width_beta", dist.Delta(width_beta_v))
            pyro.sample("x0_scale", dist.Delta(x0_scale_v))
            pyro.sample("y0_scale", dist.Delta(y0_scale_v))
            with N_plate:
                width = pyro.sample("width", dist.Delta(width_v[:,batch_idx]))
                height = pyro.sample("height", dist.Gamma(h_loc[:,batch_idx], h_beta[:,batch_idx]))
                x0 = pyro.sample("x0", dist.Normal(x_loc[:,batch_idx], x_scale[:,batch_idx]))
                y0 = pyro.sample("y0", dist.Normal(y_loc[:,batch_idx], y_scale[:,batch_idx]))


        with N_plate:
            # AoI Local Variables
            #background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx], b_beta[:,batch_idx]))
            #background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx]))
            pyro.sample("background_loc", dist.Delta(background_loc_v[:,batch_idx]))
            pyro.sample("background_beta", dist.Delta(background_beta_v[:,batch_idx]))
            with F_plate:
                # AoI & Frame Local Variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
                background = pyro.sample("background", dist.Delta(b_loc[:,batch_idx][:,:,frame_idx]))
                #background = pyro.sample("background", dist.Gamma(b_loc[:,batch_idx][:,:,frame_idx], b_beta[:,batch_idx][:,:,frame_idx]))

        return height, width, background, x0, y0
