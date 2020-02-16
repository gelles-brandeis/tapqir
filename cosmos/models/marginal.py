import torch
import torch.distributions.constraints as constraints
from pyro.infer import config_enumerate
from pyro import param

from cosmos.models.model import Model
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

    @config_enumerate
    def model(self, n_batch):
        self.model_parameters()
        data_m_pi = m_param(param("pi"), param("lamda"), self.K)  # 2**K
        control_m_pi = m_param(
            torch.tensor([1., 0.]), param("lamda"), self.K)  # 2**K
        theta_pi = theta_param(
            param("pi"), param("lamda"), self.K)  # 2**K,K+1

        self.spot_model(
            self.data, data_m_pi, theta_pi,
            self.data_mask, prefix="d")

        if self.control:
            self.spot_model(
                self.control, control_m_pi, None,
                self.control_mask, prefix="c")

    def guide(self, n_batch):
        self.guide_parameters()
        self.spot_guide(
            self.data, n_batch, theta=False, m=False,
            data_mask=self.data_mask, prefix="d")
        if self.control:
            self.spot_guide(
                self.control, n_batch, theta=False, m=False,
                data_mask=self.control_mask, prefix="c")

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+3)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        param("background_beta", torch.tensor([1.]),
              constraint=constraints.positive)
        param("height_loc", torch.tensor([1000.]),
              constraint=constraints.positive)
        param("height_beta", torch.tensor([0.01]),
              constraint=constraints.positive)
        param("width_mode", torch.tensor([1.25]),
              constraint=constraints.interval(0.5, 2.5))
        param("width_size",
              torch.tensor([10.]), constraint=constraints.positive)
        param("pi", torch.ones(2), constraint=constraints.simplex)
        param("lamda", torch.tensor([0.1]),
              constraint=constraints.interval(0., 2.))

        if self.control:
            offset_max = torch.where(
                self.data[:].min() < self.control[:].min(),
                self.data[:].min() - 0.1,
                self.control[:].min() - 0.1)
        else:
            offset_max = self.data[:].min() - 0.1
        param("offset", offset_max-50,
              constraint=constraints.interval(0, offset_max))
        param("gain", torch.tensor(5.), constraint=constraints.positive)

    def guide_parameters(self):
        # Local Parameters
        param("h_beta", torch.ones(1), constraint=constraints.positive)
        param("b_beta", torch.ones(1) * 30, constraint=constraints.positive)
        self.spot_parameters(self.data, theta=False, m=False, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, theta=False, m=False, prefix="c")
