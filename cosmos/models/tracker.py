import math
import numpy as np
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
from pyro import param 

from cosmos.models.noise import _noise, _noise_fn
from cosmos.models.model import Model
from cosmos.models.helper import Location, m_param, theta_param

class Tracker(Model):
    """ Track on-target Spot """
    def __init__(self, data, control, K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "tracker"
        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO)(max_plate_nesting=5, ignore_jit_warnings=True)
        if data.name == "Gracecy3":
            self.mcc = False
        else:
            self.mcc = True 
        super().__init__(data, control, K, lr, n_batch, jit, noise="GammaOffset")
        
    def model(self):
        data_m_pi = m_param(param("pi"), param("lamda"), self.K) # 2**K
        control_m_pi = m_param(torch.tensor([1., 0.]), param("lamda"), self.K) # 2**K
        theta_pi = theta_param(param("pi"), param("lamda"), self.K) # 2**K,K+1

        self.spot_model(self.data, data_m_pi, theta_pi, prefix="d")
    
        if self.control:
            self.spot_model(self.control, control_m_pi, None, prefix="c")

    @config_enumerate
    def guide(self):
        self.spot_guide(self.data, theta=True, prefix="d")

        if self.control:
            self.spot_guide(self.control, theta=False, prefix="c")

    def parameters(self):
        # Global Parameters
        #param("proximity", torch.tensor([(((self.D+3)/(2*0.5))**2 - 1)]), constraint=constraints.greater_than(30.))
        param("background_beta", torch.tensor([1.]), constraint=constraints.positive)
        param("height_loc", torch.tensor([1000., 1000.]), constraint=constraints.positive)
        param("height_beta", torch.tensor([1., 1.]), constraint=constraints.positive)
        param("width_mode", torch.tensor([1.3, 1.3]), constraint=constraints.interval(0.75,2.25))
        param("width_size", torch.tensor([3., 15.]), constraint=constraints.positive)
        #param("width_mode", torch.tensor([1.3]), constraint=constraints.interval(0.5,3.))
        #param("width_size", torch.tensor([10.]), constraint=constraints.positive)
        param("pi", torch.ones(2), constraint=constraints.simplex)
        param("lamda", torch.tensor([0.1]), constraint=constraints.interval(0.,2.))
        param("h_beta", torch.ones(1), constraint=constraints.positive)
        param("b_beta", torch.ones(1)*30, constraint=constraints.positive)

        if self.control:
            offset_max = torch.where(self.data[:].min() < self.control[:].min(), self.data[:].min() - 0.1, self.control[:].min() - 0.1) 
        else:
            offset_max = self.data[:].min() - 0.1 
        param("offset", offset_max-50, constraint=constraints.interval(0.,offset_max))
        param("gain", torch.tensor(5.), constraint=constraints.positive)

        # Local Parameters
        self.spot_parameters(self.data, theta=True, prefix="d")
        if self.control:
            self.spot_parameters(self.control, theta=False, prefix="c")
