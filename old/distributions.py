import math
from numbers import Number

import pyro
import torch
from torch.distributions import constraints
from pyro.distributions import TorchDistribution, Normal
from torch.distributions import Normal, Gamma, MultivariateNormal, Categorical, Uniform
from torch.distributions.utils import _standard_normal, broadcast_all

class Image(TorchDistribution):

    def __init__(self, batch_idx, frame_idx, D, z, K, b_loc, b_beta, h_loc, h_beta, w_loc, w_beta,
                 x_scale, y_scale, target_locs, validate_args=None):
        #self.data = data
        self.batch_idx, self.frame_idx, self.D = batch_idx, frame_idx, D
        self.K = K
        self.z = z
        _, self.b_loc, self.b_beta, self.h_loc, self.h_beta, self.w_loc, self.w_beta, self.x_scale, self.y_scale = broadcast_all(torch.empty(self.K-1,len(batch_idx), len(frame_idx), 1, 1), 
                        b_loc, b_beta, h_loc, h_beta, w_loc, w_beta, x_scale, y_scale)
        #self.x_scale, self.y_scale, self.target_locs, self.pixel_pos, self.spot_scale = x_scale, y_scale, target_locs, pixel_pos, spot_scale
        #self.x_scale, self.y_scale = x_scale, y_scale

        self.target_locs = target_locs
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=2).float()
        self.pixel_pos = self.pixel_pos.reshape(1,1,1,self.D,self.D,2)
        # scale matrix for gaussian spot
        self.spot_scale = torch.eye(2).reshape(1,1,1,1,1,2,2)

        #print(torch.Size([len(batch_idx), len(frame_idx)]))
        self._batch_shape = z.size()
        self._event_shape = torch.Size([self.D, self.D])

    def gaussian_spot(self, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        # select target locs for given indices
        spot_locs = self.target_locs[:,self.batch_idx][:,:,self.frame_idx] # ind,F,D,D,2
        # adjust for the center of the first frame spot_locs[...,0] += x0
        spot_locs[...,1] += y0
        rv = MultivariateNormal(spot_locs, scale_tril=self.spot_scale * width.view(width.size()+(1,1)))
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) * 2 * math.pi * width**2
        # height can be either a scalar or a vector
        # 1,K,ind,F,D,D
        return height * gaussian_spot #

    def sample(self, sample_shape=torch.Size()):
        nind, find, xind, yind = torch.meshgrid(torch.arange(len(self.batch_idx)), torch.arange(len(self.frame_idx)), torch.arange(self.D), torch.arange(self.D))
        background = Gamma(self.b_loc * self.b_beta, self.b_beta).sample()
        height = Gamma(self.h_loc * self.h_beta, self.h_beta).sample()
        width = Gamma(self.w_loc * self.w_beta, self.w_beta).sample()
        x0 = Normal(0., self.x_scale).sample()
        y0 = Normal(0., self.y_scale).sample()
        #background = pyro.sample("background", Gamma(self.b_loc * self.b_beta, self.b_beta))
        #z = pyro.sample("z", Categorical(self.pi))
        #height = pyro.sample("height", Gamma(self.h_loc * self.h_beta, self.h_beta))
        #width = pyro.sample("width", Gamma(self.w_loc * self.w_beta, self.w_beta))
        #x0 = pyro.sample("x0", Normal(0., self.x_scale))
        #y0 = pyro.sample("y0", Normal(0., self.y_scale))
        locs = torch.ones(self.K, len(self.batch_idx), len(self.frame_idx), self.D, self.D) * background
        locs += torch.cat((torch.zeros(1,len(self.batch_idx),len(self.frame_idx),self.D,self.D), self.gaussian_spot(height, width, x0, y0)), 0)
        #shape = self._extended_shape(sample_shape)
        #with torch.no_grad():
        #    return torch.normal(self.loc.expand(shape), self.scale.expand(shape))[self.z]
        #return self.gaussian_spot(height, width, x0, y0)
        return locs[self.z,nind,find,xind,yind], background, height, width, x0, y0

    def log_prob(self, value):
        locs, background, height, width, x0, y0 = value
        #if self._validate_args:
        #    self._validate_sample(value)
        # compute the variance
        #var = (self.scale ** 2)
        #log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        #return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        logs = torch.ones(self.K, len(self.batch_idx), len(self.frame_idx), self.D, self.D) * Gamma(self.b_loc * self.b_beta, self.b_beta).log_prob(background)
        logs += torch.cat((torch.zeros(1,len(self.batch_idx),len(self.frame_idx),self.D,self.D), Gamma(self.h_loc * self.h_beta, self.h_beta).log_prob(height)), 0)
        return logs[self.z] 

    def expand(self, batch_shape):
            return self
