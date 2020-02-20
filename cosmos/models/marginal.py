import torch
import torch.distributions.constraints as constraints
from pyro.infer import config_enumerate
from pyro import param
import pyro
from pyro import poutine

from cosmos.models.model import Model
from cosmos.models.hmm import HMM
from cosmos.models.helper import m_param, theta_param


class Marginal(Model):
    """ Track on-target Spot """
    def __init__(self, data, control,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "marginal"
        self.mcc = True
        super().__init__(data, control,
                         K, lr, n_batch, jit, noise="GammaOffset")

        if data.name.startswith("FL"):
            self.data_mask = torch.tensor(
                self.data.labels["spotpicker"].values < 2) \
                .reshape(self.data.N, self.data.F, 1, 1, 1).bool()
            self.control_mask = torch.tensor(
                self.control.labels["spotpicker"].values < 2) \
                .reshape(self.control.N, self.control.F, 1, 1, 1).bool()
        else:
            self.data_mask = torch.ones(
                self.data.N, self.data.F, 1, 1, 1).bool()
            if self.control:
                self.control_mask = torch.ones(
                    self.control.N, self.control.F, 1, 1, 1).bool()

    @poutine.block(hide=["width_mode", "width_size"])
    @config_enumerate
    def model(self):
        #pyro.module("cosmos", self)
        self.model_parameters()
        data_m_pi = m_param(param("pi"), param("lamda"), self.K) # 2**K
        control_m_pi = m_param(torch.tensor([1., 0.]), param("lamda"), self.K) # 2**K
        theta_pi = theta_param(param("pi"), param("lamda"), self.K) # 2**K,K+1

        self.spot_model(self.data, data_m_pi, theta_pi, self.data_mask, prefix="d")
    
        if self.control:
            self.spot_model(self.control, control_m_pi, None, self.control_mask, prefix="c")

    def guide(self):
        #pyro.module("cosmos", self)
        self.guide_parameters()
        self.spot_guide(self.data, False, False, self.data_mask, prefix="d")

        if self.control:
            self.spot_guide(self.control, False, False, self.control_mask, prefix="c")

    def guide_parameters(self):
        # Local Parameters
        self.spot_parameters(self.data, False, False, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, False, False, prefix="c")
