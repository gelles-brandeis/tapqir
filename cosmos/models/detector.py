import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.infer import config_enumerate
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
from pyro import param 

from cosmos.models.noise import _noise, _noise_fn
from cosmos.models.helper import Model, m_param, theta_param


class Detector(Model):
    """ Detect the number of Gaussian Spots """
    def __init__(self, data, control, K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "detector"
        super().__init__(data, control, K, lr, n_batch, jit, noise="GammaOffset")
        
        pyro.clear_param_store()
        self.parameters()
        self.epoch_count = 0
        self.optim = pyro.optim.Adam({"lr": lr, "betas": [0.9, 0.999]})
        self.elbo = JitTraceEnum_ELBO() if jit else TraceEnum_ELBO()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.mcc = False
    
    def model(self):
        # Constants
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])

        # Model
        offset_max = self.data._store.min() - 0.1
        offset = pyro.sample("offset", dist.Uniform(0., offset_max))
        gain = pyro.sample("gain", dist.HalfNormal(50.))

        pi = pyro.sample("pi", dist.Dirichlet(0.5 * torch.ones(self.K)))
        lamda = pyro.sample("lamda", dist.Gamma(0.01, 0.1))
        m_pi = m_param(pi, lamda, self.K) # 2**K
        c_m_pi = m_param(torch.tensor([1., 0.]), lamda, self.K) # 2**K
        #m_pi = pyro.sample("m_pi", dist.Dirichlet(torch.ones(4) / 4))

        height_loc = pyro.sample("height_loc", dist.HalfNormal(200.))
        height_beta = pyro.sample("height_beta", dist.HalfNormal(10.))
        width_mode = pyro.sample("width_mode", self.Location(1.3, 3., 0.5, 2.5))
        width_size = pyro.sample("width_size", dist.HalfNormal(100.))

        with pyro.plate("N_plate", self.data.N, subsample_size=self.n_batch, dim=-4) as batch_idx:
            with pyro.plate("F_plate", self.data.F, dim=-3):
                background = pyro.sample("background", dist.HalfNormal(1000.))
                m = pyro.sample("m", dist.Categorical(m_pi)) # N,F,1,1
                m = m_matrix[m] # N,F,1,1,K
                height = pyro.sample("height", dist.Gamma(height_loc * height_beta, height_beta).expand([len(batch_idx),self.data.F,1,1,self.K]).mask(m).to_event(1)) # K,N,F,1,1
                height = height.masked_fill(~m.bool(), 0.)
                width = pyro.sample("width", self.Location(width_mode, width_size, 0.5, 2.5).expand([len(batch_idx),self.data.F,1,1,self.K]).mask(m).to_event(1))
                x0 = pyro.sample("x0", dist.Normal(0., 10.).expand([len(batch_idx),self.data.F,1,1,self.K]).mask(m).to_event(1))
                y0 = pyro.sample("y0", dist.Normal(0., 10.).expand([len(batch_idx),self.data.F,1,1,self.K]).mask(m).to_event(1))

                spot_locs = self.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                locs = self.gaussian_spot(spot_locs, height, width, x0, y0) + background
                with pyro.plate("x_plate", size=self.D, dim=-2):
                    with pyro.plate("y_plate", size=self.D, dim=-1):
                        pyro.sample("data", self.CameraUnit(locs, gain, offset), obs=self.data[batch_idx])
    
        if self.control:
            with pyro.plate("c_N_plate", self.control.N, subsample_size=self.n_batch, dim=-4) as batch_idx:
                with pyro.plate("c_F_plate", self.control.F, dim=-3):
                    background = pyro.sample("c_background", dist.HalfNormal(1000.))
                    m = pyro.sample("c_m", dist.Categorical(c_m_pi)) # N,F,1,1
                    m = m_matrix[m] # N,F,1,1,K
                    height = pyro.sample("c_height", dist.Gamma(height_loc * height_beta, height_beta).expand([len(batch_idx),self.control.F,1,1,self.K]).mask(m).to_event(1)) # K,N,F,1,1
                    height = height.masked_fill(~m.bool(), 0.)
                    width = pyro.sample("c_width", self.Location(width_mode, width_size, 0.5, 2.5).expand([len(batch_idx),self.control.F,1,1,self.K]).mask(m).to_event(1))
                    x0 = pyro.sample("c_x0", dist.Normal(0., 10.).expand([len(batch_idx),self.control.F,1,1,self.K]).mask(m).to_event(1))
                    y0 = pyro.sample("c_y0", dist.Normal(0., 10.).expand([len(batch_idx),self.control.F,1,1,self.K]).mask(m).to_event(1))

                    spot_locs = self.control_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                    locs = self.gaussian_spot(spot_locs, height, width, x0, y0) + background
                    with pyro.plate("c_x_plate", size=self.D, dim=-2):
                        with pyro.plate("c_y_plate", size=self.D, dim=-1):
                            pyro.sample("c_data", self.CameraUnit(locs, gain, offset), obs=self.control[batch_idx])

    def guide(self):
        # Constants
        m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])

        # Guide
        pyro.sample("offset", dist.Delta(param("offset_v")))
        pyro.sample("gain", dist.Delta(param("gain_v")))

        pyro.sample("pi", dist.Dirichlet(param("pi_concentration")))
        pyro.sample("lamda", dist.Gamma(param("lamda_loc") * param("lamda_beta"), param("lamda_beta")))
        #pyro.sample("m_pi", dist.Dirichlet(param("m_pi_concentration")))

        pyro.sample("height_loc", dist.Delta(param("height_loc_v")))
        pyro.sample("height_beta", dist.Delta(param("height_beta_v")))
        pyro.sample("width_mode", dist.Delta(param("width_mode_v")))
        pyro.sample("width_size", dist.Delta(param("width_size_v")))

        with pyro.plate("N_plate", self.data.N, subsample_size=self.n_batch, dim=-4) as batch_idx:
            with pyro.plate("F_plate", self.data.F, dim=-3):
                pyro.sample("background", dist.Gamma(param("b_loc")[batch_idx] * param("b_beta"), param("b_beta")))
                m = pyro.sample("m", dist.Categorical(param("m_probs")[batch_idx]), infer={"enumerate": "sequential"})
                m = m_matrix[m] # N,F,1,1,K
                pyro.sample("height", dist.Gamma(param("h_loc")[batch_idx] * param("h_beta"), param("h_beta")).mask(m).to_event(1))
                pyro.sample("width", self.Location(param("w_mode")[batch_idx], param("w_size")[batch_idx], 0.5, 2.5).mask(m).to_event(1))
                pyro.sample("x0", dist.Normal(param("x_mean")[batch_idx], param("scale")[batch_idx]).mask(m).to_event(1))
                pyro.sample("y0", dist.Normal(param("y_mean")[batch_idx], param("scale")[batch_idx]).mask(m).to_event(1))

        if self.control:
            with pyro.plate("c_N_plate", self.control.N, subsample_size=self.n_batch, dim=-4) as batch_idx:
                with pyro.plate("c_F_plate", self.control.F, dim=-3):
                    pyro.sample("c_background", dist.Gamma(param("c_b_loc")[batch_idx] * param("b_beta"), param("b_beta")))
                    m = pyro.sample("c_m", dist.Categorical(param("c_m_probs")[batch_idx]), infer={"enumerate": "sequential"})
                    m = m_matrix[m] # N,F,1,1,K
                    pyro.sample("c_height", dist.Gamma(param("c_h_loc")[batch_idx] * param("h_beta"), param("h_beta")).mask(m).to_event(1))
                    pyro.sample("c_width", self.Location(param("c_w_mode")[batch_idx], param("c_w_size")[batch_idx], 0.5, 2.5).mask(m).to_event(1))
                    pyro.sample("c_x0", dist.Normal(param("c_x_mean")[batch_idx], param("c_scale")[batch_idx]).mask(m).to_event(1))
                    pyro.sample("c_y0", dist.Normal(param("c_y_mean")[batch_idx], param("c_scale")[batch_idx]).mask(m).to_event(1))

    def parameters(self):
        pyro.get_param_store().load(os.path.join(self.data.path, "runs", self.data.name, "features/K{}".format(self.K), "lr0.001", "params"))

        param("height_loc_v", torch.tensor([200.]), constraint=constraints.positive)
        param("height_beta_v", torch.tensor([1.]), constraint=constraints.positive)
        param("width_mode_v", torch.tensor([1.3]), constraint=constraints.positive)
        param("width_size_v", torch.tensor([100.]), constraint=constraints.positive)
        #param("m_pi_concentration", torch.ones(4)*self.data.N*self.data.F/4, constraint=constraints.positive)
        param("pi_concentration", torch.ones(2)*self.data.N*self.data.F/2, constraint=constraints.positive)
        param("lamda_loc", torch.tensor([0.1]), constraint=constraints.positive)
        param("lamda_beta", torch.tensor([10.]), constraint=constraints.positive)

        # Data
        h_max = np.percentile(param("h_loc").detach().cpu(), 95)
        p = torch.where(param("h_loc").detach() < h_max, param("h_loc").detach()/h_max, torch.tensor(1.))

        m3 = 0.9 * p[...,0] * p[...,1] + 0.025
        m2 = 0.9 * (1 - p[...,0]) * p[...,1] + 0.025
        m1 = 0.9 * p[...,0] * (1 - p[...,1]) + 0.025
        m0 = 0.9 * (1 - p[...,0]) * (1 - p[...,1]) + 0.025

        m_probs = torch.stack((m0,m1,m2,m3), dim=-1)
        param("m_probs", m_probs.reshape(self.data.N,self.data.F,1,1,4), constraint=constraints.simplex)

        # Control
        if self.control:
            h_max = np.percentile(param("c_h_loc").detach().cpu(), 95)
            p = torch.where(param("c_h_loc").detach() < h_max, param("c_h_loc").detach()/h_max, torch.tensor(1.))

            m3 = 0.9 * p[...,0] * p[...,1] + 0.025
            m2 = 0.9 * (1 - p[...,0]) * p[...,1] + 0.025
            m1 = 0.9 * p[...,0] * (1 - p[...,1]) + 0.025
            m0 = 0.9 * (1 - p[...,0]) * (1 - p[...,1]) + 0.025

            m_probs = torch.stack((m0,m1,m2,m3), dim=-1)
            param("c_m_probs", m_probs.reshape(self.control.N,self.control.F,1,1,4), constraint=constraints.simplex)
            param("c_m_pi_concentration", torch.ones(4)*self.control.N*self.control.F/4, constraint=constraints.positive)
