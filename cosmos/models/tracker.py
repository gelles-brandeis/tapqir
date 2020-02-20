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

        # self.param_path = os.path.join(
        #    self.data.path, "runs", "{}".format(self.data.name),
        #    "marginalnn", "K{}".format(self.K),
        #    "{}".format("jit" if jit else "nojit"),
        #    "lr0.005", "{}".format(self.optim_fn.__name__),
        #    "{}".format(self.n_batch))
        # pyro.get_param_store().load(
        #    os.path.join(self.param_path, "params"),
        #    map_location=self.data.device)

    @poutine.block(hide=["width_mode", "width_size"])
    def model(self):
        self.model_parameters()
        data_m_pi = m_param(param("pi"), param("lamda"), self.K)
        control_m_pi = m_param(torch.tensor([1., 0.]), param("lamda"), self.K)
        theta_pi = theta_param(param("pi"), param("lamda"), self.K)

        with scope(prefix="d"):
            self.spot_model(
                self.data, data_m_pi, theta_pi,
                self.data_mask, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_model(
                    self.control, control_m_pi, None,
                    self.control_mask, prefix="c")

    @config_enumerate
    def guide(self):
        self.guide_parameters()
        with scope(prefix="d"):
            self.spot_guide(
                self.data, True, self.data_mask, prefix="d")

        if self.control:
            with scope(prefix="c"):
                self.spot_guide(
                    self.control, False, self.control_mask, prefix="c")

    def guide_parameters(self):
        self.spot_parameters(self.data, True, True, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, True, False, prefix="c")
