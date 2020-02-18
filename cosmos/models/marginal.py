import torch
import torch.distributions.constraints as constraints
from pyro.infer import config_enumerate
from pyro import param
import pyro
from pyro import poutine

from cosmos.models.model import Model
from cosmos.models.hmm import HMM
from cosmos.models.helper import j_param


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

    #@poutine.block(hide=["width_mode", "width_size"])
    @config_enumerate
    def model(self):
        #pyro.module("cosmos", self)
        self.model_parameters()
        j_pi = j_param(param("j_pi"), self.K)  # 2**K
        theta_pi = torch.tensor([[1., 0., 0.], [0., 0.5, 0.5]])

        self.spot_model(
            self.data, True, theta_pi, j_pi,
            self.data_mask, prefix="d")

        if self.control:
            self.spot_model(
                self.control, False, theta_pi, j_pi,
                self.control_mask, prefix="c")

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
            self.spot_parameters(self.control, False, False, prefix="c")
