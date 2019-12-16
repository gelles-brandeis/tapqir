import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.infer import JitTrace_ELBO, Trace_ELBO
import pyro.distributions as dist
from pyro import param 

from cosmos.models.noise import _noise, _noise_fn
from cosmos.models.helper import Model 


class Features(Model):
    """ Extract features of the Gaussian Spot Model """
    def __init__(self, data, control, K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "features"
        self.elbo = JitTrace_ELBO() if jit else Trace_ELBO()
        self.mcc = False
        super().__init__(data, control, K, lr, n_batch, jit, noise="GammaOffset")
    
    def model(self):
        # Model 
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(0., offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(50.))
        
        with pyro.plate("N_plate", self.data.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
            with pyro.plate("F_plate", self.data.F, dim=-4):
                background = pyro.sample("background", dist.HalfNormal(1000.))
                with pyro.plate("K_plate", self.K, dim=-1):
                    height = pyro.sample("height", dist.HalfNormal(3000.)) # N,F,1,1,K
                    width = pyro.sample("width", self.Location(1.3, 10., 0.5, 2.5))
                    x0 = pyro.sample("x0", dist.Normal(0., 10.))
                    y0 = pyro.sample("y0", dist.Normal(0., 10.))

                spot_locs = self.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                locs = self.gaussian_spot(spot_locs, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-3):
                    with pyro.plate("y_plate", size=self.D, dim=-2):
                        pyro.sample("data", self.CameraUnit(locs, gain, offset), obs=self.data[batch_idx].unsqueeze(dim=-1))
    
        if self.control:
            with pyro.plate("c_N_plate", self.control.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
                with pyro.plate("c_F_plate", self.control.F, dim=-4):
                    background = pyro.sample("c_background", dist.HalfNormal(1000.))
                    with pyro.plate("c_K_plate", self.K, dim=-1):
                        height = pyro.sample("c_height", dist.HalfNormal(3000.)) # N,F,1,1,K
                        width = pyro.sample("c_width", self.Location(1.3, 10., 0.5, 2.5))
                        x0 = pyro.sample("c_x0", dist.Normal(0., 10.))
                        y0 = pyro.sample("c_y0", dist.Normal(0., 10.))

                    spot_locs = self.control_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                    locs = self.gaussian_spot(spot_locs, height, width, x0, y0) + background
                    with pyro.plate("c_x_plate", size=self.D, dim=-3):
                        with pyro.plate("c_y_plate", size=self.D, dim=-2):
                            pyro.sample("c_data", self.CameraUnit(locs, gain, offset), obs=self.control[batch_idx].unsqueeze(dim=-1))

    def guide(self):
        # Guide
        pyro.sample("offset", dist.Delta(param("offset_v")))
        pyro.sample("gain", dist.Delta(param("gain_v")))

        with pyro.plate("N_plate", self.data.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
            with pyro.plate("F_plate", self.data.F, dim=-4):
                pyro.sample("background", dist.Gamma(param("b_loc")[batch_idx] * param("b_beta"), param("b_beta")))
                with pyro.plate("K_plate", self.K, dim=-1):
                    pyro.sample("height", dist.Gamma(param("h_loc")[batch_idx] * param("h_beta"), param("h_beta")))
                    pyro.sample("width", self.Location(param("w_mode")[batch_idx], param("w_size")[batch_idx], 0.5, 2.5))
                    pyro.sample("x0", dist.Normal(param("x_mean")[batch_idx], param("scale")[batch_idx]))
                    pyro.sample("y0", dist.Normal(param("y_mean")[batch_idx], param("scale")[batch_idx]))

        if self.control:
            with pyro.plate("c_N_plate", self.control.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
                with pyro.plate("c_F_plate", self.control.F, dim=-4):
                    pyro.sample("c_background", dist.Gamma(param("c_b_loc")[batch_idx] * param("b_beta"), param("b_beta")))
                    with pyro.plate("c_K_plate", self.K, dim=-1):
                        pyro.sample("c_height", dist.Gamma(param("c_h_loc")[batch_idx] * param("h_beta"), param("h_beta")))
                        pyro.sample("c_width", self.Location(param("c_w_mode")[batch_idx], param("c_w_size")[batch_idx], 0.5, 2.5))
                        pyro.sample("c_x0", dist.Normal(param("c_x_mean")[batch_idx], param("c_scale")[batch_idx]))
                        pyro.sample("c_y0", dist.Normal(param("c_y_mean")[batch_idx], param("c_scale")[batch_idx]))

    def parameters(self):
        # Parameters
        if self.control:
            offset_max = torch.where(self.data._store.min() < self.control._store.min(), self.data._store.min() - 0.1, self.control._store.min() - 0.1) 
        else:
            offset_max = self.data._store.min() - 0.1 
        param("offset_v", offset_max-50, constraint=constraints.interval(0.,offset_max))
        param("gain_v", torch.tensor(5.), constraint=constraints.positive)

        param("b_loc", torch.ones(self.data.N,self.data.F,1,1,1)*10., constraint=constraints.positive)
        param("b_beta", torch.ones(1)*self.D**2, constraint=constraints.positive)
        intensity = torch.ones(self.data.N,self.data.F,1,1,self.K)*10.
        intensity[...,1] = 30. # 30 is better than 100
        param("h_loc", intensity, constraint=constraints.positive)
        param("h_beta", torch.ones(1), constraint=constraints.positive)
        param("w_mode", torch.ones(self.data.N,self.data.F,1,1,self.K)*1.35, constraint=constraints.interval(0.5,3.))
        param("w_size", torch.ones(self.data.N,self.data.F,1,1,self.K)*100., constraint=constraints.greater_than(2.))
        param("x_mean", torch.zeros(self.data.N,self.data.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        param("y_mean", torch.zeros(self.data.N,self.data.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        scale = torch.sqrt((param("w_mode")**2 + 1/12) / param("h_loc") + 8 * math.pi * param("w_mode")**4 * param("b_loc") / param("h_loc")**2)
        param("scale", scale, constraint=constraints.positive)
        
        if self.control:
            param("c_b_loc", torch.ones(self.control.N,self.control.F,1,1,1)*10., constraint=constraints.positive)
            intensity = torch.ones(self.control.N,self.control.F,1,1,self.K)*10.
            intensity[...,1] = 30. # 30 is better than 100
            param("c_h_loc", intensity, constraint=constraints.positive)
            param("c_w_mode", torch.ones(self.control.N,self.control.F,1,1,self.K)*1.35, constraint=constraints.interval(0.5,3.))
            param("c_w_size", torch.ones(self.control.N,self.control.F,1,1,self.K)*100., constraint=constraints.greater_than(2.))
            param("c_x_mean", torch.zeros(self.control.N,self.control.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
            param("c_y_mean", torch.zeros(self.control.N,self.control.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
            scale = torch.sqrt((param("c_w_mode")**2 + 1/12) / param("c_h_loc") + 8 * math.pi * param("c_w_mode")**4 * param("c_b_loc") / param("c_h_loc")**2)
            param("c_scale", scale, constraint=constraints.positive)
