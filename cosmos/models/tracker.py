import torch
from pyro.infer import config_enumerate
from pyro import param
from pyro import poutine
from pyro.contrib.autoname import scope

from cosmos.models.model import Model
from cosmos.models.helper import m_param, theta_param


class Tracker(Model):
    """ Track on-target Spot """
    def __init__(self, data, control,
                 K, lr, n_batch, jit, noise="GammaOffset"):
        self.__name__ = "tracker"
        if data.name == "Gracecy3":
            self.mcc = False
        else:
            self.mcc = True
        super().__init__(data, control,
                         K, lr, n_batch, jit, noise="GammaOffset")

    # @poutine.block(hide=["width_mode", "width_size"])
    def model(self):
        self.model_parameters()
        data_m_pi = m_param(param("pi"), param("lamda"), self.K)
        control_m_pi = m_param(torch.tensor([1., 0.]), param("lamda"), self.K)
        theta_pi = theta_param(param("pi"), param("lamda"), self.K)

        with scope(prefix="d"):
            self.spot_model(self.data, data_m_pi, theta_pi, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_model(self.control, control_m_pi, None, prefix="c")

    @config_enumerate
    def guide(self):
        self.guide_parameters()
        with scope(prefix="d"):
            self.spot_guide(self.data, True, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_guide(self.control, False, prefix="c")

    def guide_parameters(self):
        self.spot_parameters(self.data, True, True, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, True, False, prefix="c")
