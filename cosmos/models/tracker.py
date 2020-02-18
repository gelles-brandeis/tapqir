import torch
import torch.distributions.constraints as constraints
from pyro.infer import config_enumerate
from pyro import param
from pyro import poutine
import os
import pyro

from cosmos.models.model import Model
from cosmos.models.helper import j_param


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

        self.param_path = os.path.join(
            self.data.path, "runs", "{}".format(self.data.name),
            "marginalnn", "K{}".format(self.K),
            "{}".format("jit" if jit else "nojit"),
            "lr0.005", "{}".format(self.optim_fn.__name__),
            "{}".format(self.n_batch))
        pyro.get_param_store().load(
            os.path.join(self.param_path, "params"),
            map_location=self.data.device)

    @poutine.block(hide_types=["param"])
    def model(self):
        self.model_parameters()
        j_pi = j_param(param("lamda"), self.K)  # 2**K

        self.spot_model(
            self.data, True, j_pi,
            self.data_mask, prefix="d")

        if self.control:
            self.spot_model(
                self.control, False, j_pi,
                self.control_mask, prefix="c")

    @poutine.block(expose_types=["sample"], expose=["d/z_probs", "d/j_probs", "c/j_probs"])
    @config_enumerate
    def guide(self):
        self.guide_parameters()
        self.spot_guide(
            self.data, True, True,
            data_mask=self.data_mask, prefix="d")
        if self.control:
            self.spot_guide(
                self.control, False, True,
                data_mask=self.control_mask, prefix="c")

    def guide_parameters(self):
        # Local Parameters
        self.spot_parameters(self.data, True, True, prefix="d")
        if self.control:
            self.spot_parameters(
                self.control, False, True, prefix="c")
