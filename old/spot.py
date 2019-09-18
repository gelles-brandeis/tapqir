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


class GaussianSpot:
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
        self.target_locs = self.target_locs.reshape(1,self.N,self.F,1,1,2)
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
    
    def locs_mixture_model(self, batch_idx, frame_idx):
        #plates
        #J_plate = pyro.plate("junk_axis", 1, dim=-6)
        #K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        # weights
        weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        
        nind, find, xind, yind = torch.tensor(np.indices((len(batch_idx),len(frame_idx),self.D,self.D)))
        
        # global class and locs variables
        background_loc = pyro.sample("background_loc", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
        background_scale = pyro.sample("background_scale", dist.HalfNormal(100.).expand([1,1,1,1,1]))
        
        
        #with K_plate:
        height_loc = pyro.sample("height_loc", dist.HalfNormal(500.).expand([self.K-1,1,1,1,1]))
        height_scale = pyro.sample("height_scale", dist.HalfNormal(50.).expand([self.K-1,1,1,1,1]))
        width_alpha = pyro.sample("width_alpha", dist.HalfNormal(50.).expand([self.K-1,1,1,1,1]))
        width_beta = pyro.sample("width_beta", dist.HalfNormal(5.).expand([self.K-1,1,1,1,1]))
        x0_scale = pyro.sample("x0_scale", dist.HalfNormal(5.).expand([self.K-1,1,1,1,1]))
        y0_scale = pyro.sample("y0_scale", dist.HalfNormal(5.).expand([self.K-1,1,1,1,1]))
        
        with N_plate:
            background = pyro.sample("background", dist.Normal(background_loc, background_scale))
            height = pyro.sample("height", dist.Normal(height_loc, height_scale))
            width = pyro.sample("width", dist.Gamma(width_alpha, width_beta))
            x0 = pyro.sample("x0", dist.Normal(0., x0_scale))
            y0 = pyro.sample("y0", dist.Normal(0., y0_scale))
            # 1,1,N,1,1,1
            with F_plate:
                #background = pyro.sample("background", dist.Normal(background_loc, background_scale))
                # N,F,1,1
                z = pyro.sample("z", dist.Categorical(weights))
            
        # return locs for K classes
        locs = torch.ones(self.K, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(batch_idx),len(frame_idx),self.D,self.D), self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0)), 0)

        return locs[z,nind,find,xind,yind]
    
    def locs_mixture_guide(self, batch_idx, frame_idx):
        # plates
        #K_plate = pyro.plate("component_axis", self.K-1, dim=-5)
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        
        weights_conc = pyro.param("weights_conc", torch.ones(self.K)*self.N*self.F/self.K, constraint=constraints.positive)
        
        background_loc_delta = pyro.param("background_loc_delta", torch.tensor(60.), constraint=constraints.positive)
        background_scale_delta = pyro.param("background_scale_delta", torch.tensor(1.), constraint=constraints.positive)
        
        z_probs = pyro.param("z_probs", torch.ones(self.N,self.F,1,1,self.K) / self.K, constraint=constraints.simplex)

        height_loc_delta = pyro.param("height_loc_delta", torch.ones(self.K-1)*100, constraint=constraints.positive)
        height_scale_delta = pyro.param("height_scale_delta", torch.ones(self.K-1), constraint=constraints.positive)
        width_alpha_delta = pyro.param("width_alpha_delta", torch.ones(self.K-1)*1.5, constraint=constraints.positive)
        width_beta_delta = pyro.param("width_beta_delta", torch.ones(self.K-1)*0.3, constraint=constraints.positive)
        #x0_loc_delta = pyro.param("x0_loc_delta", torch.zeros(self.K-1), constraint=constraints.interval(-10, 10))
        x0_scale_delta = pyro.param("x0_scale_delta", torch.ones(self.K-1)*0.5, constraint=constraints.positive)
        #y0_loc_delta = pyro.param("y0_loc_delta", torch.zeros(self.K-1), constraint=constraints.interval(-10, 10))
        y0_scale_delta = pyro.param("y0_scale_delta", torch.ones(self.K-1)*0.5, constraint=constraints.positive)
        
        # local
        background_delta = pyro.param("background_delta", torch.ones(1,self.N,1,1,1)*60, constraint=constraints.positive)
        #background_delta = pyro.param("background_delta", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        height_delta = pyro.param("height_delta", torch.ones(self.K-1,self.N,1,1,1)*100, constraint=constraints.positive)
        width_delta = pyro.param("width_delta", torch.ones(self.K-1,self.N,1,1,1)*1.5, constraint=constraints.positive)
        x0_delta = pyro.param("x_delta", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        y0_delta = pyro.param("y_delta", torch.zeros(self.K-1,self.N,1,1,1), constraint=constraints.real)
        
        pyro.sample("weights", dist.Dirichlet(weights_conc))
        #pyro.sample("junk_weights", dist.Dirichlet(junk_weights_conc))
        
        pyro.sample("background_loc", dist.Delta(background_loc_delta))
        pyro.sample("background_scale", dist.Delta(background_scale_delta))
            
                
        #with K_plate:
        pyro.sample("height_loc", dist.Delta(height_loc_delta).expand([self.K-1,1,1,1,1]))
        pyro.sample("height_scale", dist.Delta(height_scale_delta).expand([self.K-1,1,1,1,1]))
        pyro.sample("width_alpha", dist.Delta(width_alpha_delta).expand([self.K-1,1,1,1,1]))
        pyro.sample("width_beta", dist.Delta(width_beta_delta).expand([self.K-1,1,1,1,1]))
        #x0_loc = pyro.sample("x0_loc", dist.Delta(x0_loc_delta))
        pyro.sample("x0_scale", dist.Delta(x0_scale_delta).expand([self.K-1,1,1,1,1]))
        #y0_loc = pyro.sample("y0_loc", dist.Delta(y0_loc_delta))
        pyro.sample("y0_scale", dist.Delta(y0_scale_delta).expand([self.K-1,1,1,1,1]))
        
        with N_plate:
            background = pyro.sample("background", dist.Delta(background_delta[:,batch_idx]))
            height = pyro.sample("height", dist.Delta(height_delta[:,batch_idx]))
            width = pyro.sample("width", dist.Delta(width_delta[:,batch_idx]))
            x0 = pyro.sample("x0", dist.Delta(x0_delta[:,batch_idx]))
            y0 = pyro.sample("y0", dist.Delta(y0_delta[:,batch_idx]))
                        
            with F_plate:
                #background = pyro.sample("background", dist.Delta(background_delta[:,batch_idx][:,:,frame_idx]))
                # local hidden variables
                pyro.sample("z", dist.Categorical(z_probs[batch_idx][:,frame_idx]))
        
        return height, width, background, x0, y0


    def locs_feature_model(self, batch_idx, frame_idx):
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        
        #weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(self.K)))
        
        # global locs variables
        width = pyro.sample("width", dist.Gamma(1, 0.1).expand([1,1,1,1,1]))
        with N_plate:
            #background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
            with F_plate:
                #z = pyro.sample("z", dist.Categorical(weights))
                # local height and locs
                background = pyro.sample("background", dist.HalfNormal(1000.).expand([1,1,1,1,1]))
                height = pyro.sample("height", dist.HalfNormal(500.).expand([1,1,1,1,1]))
                x0 = pyro.sample("x0", dist.Normal(0.,1.).expand([1,1,1,1,1]))
                y0 = pyro.sample("y0", dist.Normal(0.,1.).expand([1,1,1,1,1]))
        #locs = torch.ones(1, len(batch_idx), len(frame_idx), self.D, self.D) * background
        locs = self.gaussian_spot(batch_idx, frame_idx, height, width, x0, y0) + background
        #locs = self.gaussian_spot(torch.arange(len(self.data)), height, width, x0, y0) + background
        #print(background.shape, x0.shape, locs.shape)
        return locs[0]


    def locs_feature_guide(self, batch_idx, frame_idx):
        N_plate = pyro.plate("sample_axis", self.N, subsample=batch_idx, dim=-4)
        F_plate = pyro.plate("frame_axis", size=self.F, subsample=frame_idx, dim=-3)
        
        # global locs variables
        auto_background = pyro.param("auto_background", self.data.background.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        
        auto_width = pyro.param("auto_width", torch.ones(1,1,1,1,1)*1.5, constraint=constraints.interval(0,5))
        
        # local variables
        auto_x0 = pyro.param("auto_x0", torch.zeros(1,self.N,self.F,1,1), constraint=constraints.real)
        auto_y0 = pyro.param("auto_y0", torch.zeros(1,self.N,self.F,1,1), constraint=constraints.real)
        auto_height = pyro.param("auto_height", self.data.height.reshape(1,self.N,self.F,1,1), constraint=constraints.positive)
        
        pyro.sample("width", dist.Delta(auto_width))
        with N_plate:
            #pyro.sample("background", dist.Delta(background_delta[:,batch_idx]))
            with F_plate:
                # local height and locs
                pyro.sample("background", dist.Delta(auto_background[:,batch_idx][:,:,frame_idx]))
                height = pyro.sample("height", dist.Delta(auto_height[:,batch_idx][:,:,frame_idx]))
                x0 = pyro.sample("x0", dist.Delta(auto_x0[:,batch_idx][:,:,frame_idx]))
                y0 = pyro.sample("y0", dist.Delta(auto_y0[:,batch_idx][:,:,frame_idx]))
        
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
        