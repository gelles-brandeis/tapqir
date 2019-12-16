import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
import pyro.distributions as dist
from pyro import param 

from cosmos.models.noise import _noise, _noise_fn
from cosmos.models.model import Model
from cosmos.models.helper import Location, m_param, theta_param

class Tracker(Model):
    """ Track on-target Spot """
    def __init__(self, data, control, K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "tracker"
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.mcc = True 
        super().__init__(data, control, K, lr, n_batch, jit, noise="GammaOffset")
        
    def model(self):
        # Constants
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])
        theta_matrix = torch.tensor([[0,0], [1,0], [0,1]]) # K+1,K
        scale = torch.tensor([10., 0.5])

        # Model
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(0., offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(50.))

        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        lamda = pyro.sample("lamda", dist.Gamma(0.01, 0.1))
        m_pi = m_param(pi, lamda, self.K) # 2**K
        c_m_pi = m_param(torch.tensor([1., 0.]), lamda, self.K) # 2**K
        theta_pi = theta_param(pi, lamda, self.K) # 2**K,K+1

        height_loc = pyro.sample("height_loc", dist.HalfNormal(200.))
        height_beta = pyro.sample("height_beta", dist.HalfNormal(10.))
        width_mode = pyro.sample("width_mode", Location(1.3, 3., 0.5, 2.5))
        width_size = pyro.sample("width_size", dist.HalfNormal(100.))

        with pyro.plate("N_plate", self.data.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
            with pyro.plate("F_plate", self.data.F, dim=-4):
                background = pyro.sample("background", dist.HalfNormal(1000.))
                m = pyro.sample("m", dist.Categorical(m_pi)) # N,F,1,1,1
                theta = pyro.sample("theta", dist.Categorical(theta_pi[m])) # N,F,1,1,1
                m = m_matrix[m.squeeze(dim=-1)] # N,F,1,1,K
                theta = theta_matrix[theta.squeeze(dim=-1)] # N,F,1,1,K   K+1,1,1,1,1,1
                with pyro.plate("K_plate", self.K, dim=-1):
                    with pyro.poutine.mask(mask=m.byte()):
                        height = pyro.sample("height", dist.Gamma(height_loc * height_beta, height_beta)) # K,N,F,1,1
                        height = height.masked_fill(~m.bool(), 0.)
                        width = pyro.sample("width", Location(width_mode, width_size, 0.5, 2.5))
                        x0 = pyro.sample("x0", dist.Normal(0., scale[theta]))
                        y0 = pyro.sample("y0", dist.Normal(0., scale[theta]))

                spot_locs = self.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                locs = self.gaussian_spot(spot_locs, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-3):
                    with pyro.plate("y_plate", size=self.D, dim=-2):
                        pyro.sample("data", self.CameraUnit(locs, gain, offset), obs=self.data[batch_idx].unsqueeze(dim=-1))
    
        if self.control:
            with pyro.plate("c_N_plate", self.control.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
                with pyro.plate("c_F_plate", self.control.F, dim=-4):
                    background = pyro.sample("c_background", dist.HalfNormal(1000.))
                    m = pyro.sample("c_m", dist.Categorical(c_m_pi)) # N,F,1,1
                    m = m_matrix[m.squeeze(dim=-1)] # N,F,1,1,K
                    with pyro.plate("c_K_plate", self.K, dim=-1):
                        with pyro.poutine.mask(mask=m.byte()):
                            height = pyro.sample("c_height", dist.Gamma(height_loc * height_beta, height_beta)) # K,N,F,1,1
                            height = height.masked_fill(~m.bool(), 0.)
                            width = pyro.sample("c_width", Location(width_mode, width_size, 0.5, 2.5))
                            x0 = pyro.sample("c_x0", dist.Normal(0., 10.))
                            y0 = pyro.sample("c_y0", dist.Normal(0., 10.))

                    spot_locs = self.control_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                    locs = self.gaussian_spot(spot_locs, height, width, x0, y0) + background
                    with pyro.plate("c_x_plate", size=self.D, dim=-3):
                        with pyro.plate("c_y_plate", size=self.D, dim=-2):
                            pyro.sample("c_data", self.CameraUnit(locs, gain, offset), obs=self.control[batch_idx].unsqueeze(dim=-1))

    def guide(self):
        # Constants
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])

        # Guide
        pyro.sample("offset", dist.Delta(param("offset_v")))
        pyro.sample("gain", dist.Delta(param("gain_v")))

        pyro.sample("pi", dist.Dirichlet(param("pi_concentration")))
        pyro.sample("lamda", dist.Gamma(param("lamda_loc") * param("lamda_beta"), param("lamda_beta")))

        pyro.sample("height_loc", dist.Delta(param("height_loc_v")))
        pyro.sample("height_beta", dist.Delta(param("height_beta_v")))
        pyro.sample("width_mode", dist.Delta(param("width_mode_v")))
        pyro.sample("width_size", dist.Delta(param("width_size_v")))

        with pyro.plate("N_plate", self.data.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
            with pyro.plate("F_plate", self.data.F, dim=-4):
                pyro.sample("background", dist.Gamma(param("b_loc")[batch_idx] * param("b_beta"), param("b_beta")))
                m = pyro.sample("m", dist.Categorical(param("m_probs")[batch_idx]), infer={"enumerate": "parallel"})
                m = m_matrix[m.squeeze(dim=-1)] # N,F,1,1,K
                pyro.sample("theta", dist.Categorical(param("theta_probs")[batch_idx]), infer={"enumerate": "parallel"}) # N,F,1,1
                with pyro.plate("K_plate", self.K, dim=-1):
                    with pyro.poutine.mask(mask=m.byte()):
                        pyro.sample("height", dist.Gamma(param("h_loc")[batch_idx] * param("h_beta"), param("h_beta")))
                        pyro.sample("width", Location(param("w_mode")[batch_idx], param("w_size")[batch_idx], 0.5, 2.5))
                        pyro.sample("x0", dist.Normal(param("x_mean")[batch_idx], param("scale")[batch_idx]))
                        pyro.sample("y0", dist.Normal(param("y_mean")[batch_idx], param("scale")[batch_idx]))

        if self.control:
            with pyro.plate("c_N_plate", self.control.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
                with pyro.plate("c_F_plate", self.control.F, dim=-4):
                    pyro.sample("c_background", dist.Gamma(param("c_b_loc")[batch_idx] * param("b_beta"), param("b_beta")))
                    m = pyro.sample("c_m", dist.Categorical(param("c_m_probs")[batch_idx]), infer={"enumerate": "parallel"})
                    m = m_matrix[m.squeeze(dim=-1)] # N,F,1,1,K
                    with pyro.plate("c_K_plate", self.K, dim=-1):
                        with pyro.poutine.mask(mask=m.byte()):
                            pyro.sample("c_height", dist.Gamma(param("c_h_loc")[batch_idx] * param("h_beta"), param("h_beta")))
                            pyro.sample("c_width", Location(param("c_w_mode")[batch_idx], param("c_w_size")[batch_idx], 0.5, 2.5))
                            pyro.sample("c_x0", dist.Normal(param("c_x_mean")[batch_idx], param("c_scale")[batch_idx]))
                            pyro.sample("c_y0", dist.Normal(param("c_y_mean")[batch_idx], param("c_scale")[batch_idx]))

    def parameters(self):
        pyro.get_param_store().load(os.path.join(self.data.path, "runs", self.data.name, "detector/K{}".format(self.K), "lr0.001", "params"))
        param("theta_probs", torch.ones(self.data.N,self.data.F,1,1,1,self.K+1), constraint=constraints.simplex)
