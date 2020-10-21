import torch
from torch.distributions import constraints
from pyro.distributions import TorchDistribution, Gamma
from pykeops.torch import Genred
import math


class MaskedNormalKDE(TorchDistribution):
    r"""
    Sum of the offset variable and the Gamma distributed variable::
    """

    arg_constraints = {}

    def __init__(self, mask, loc, beta, logits0, logits1, device="GPU"):
        self.dtype = "float32"
        self.device = device
        self._mask = mask
        self.loc = loc.reshape(-1, 1)
        self.var = (loc / beta).reshape(-1, 1)
        self.logits0 = logits0.reshape(-1, 1)
        self.logits1 = logits1.reshape(-1, 1)
        if self.device == "GPU":
            self.p = torch.tensor([math.log(math.sqrt(2 * math.pi))]).cuda()
        else:
            self.p = torch.tensor([math.log(math.sqrt(2 * math.pi))])
        #self.dist = Gamma(concentration, rate)
        # batch_shape = loc.shape[:-1]
        batch_shape = torch.Size([])
        event_shape = torch.Size([])
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        shape = value.shape
        value = value.reshape(-1, 1)
        formula0 = 'm-Square(x-loc)/(IntCst(2)*var)-Log(Sqrt(var))-p'
        variables0 = ['x = Vi(1)',  # First arg   : i-variable, of size 1 (scalar)
                     'm = Vj(1)',  # Second arg  : j-variable, of size 1 (scalar)
                     'loc = Vj(1)',  # Second arg  : j-variable, of size 1 (scalar)
                     'var = Vj(1)',
                     'p = Pm(1)']

        my_routine0 = Genred(formula0, variables0, reduction_op='LogSumExp', axis=1, dtype=self.dtype)
        result0 = my_routine0(value, self.logits0, self.loc, self.var, self.p, backend=self.device)
        result0 = result0.reshape(shape)

        formula1 = 'm-Square(x-loc)/(IntCst(2)*var)-Log(Sqrt(var))-p'
        variables1 = ['x = Vi(1)',  # First arg   : i-variable, of size 1 (scalar)
                     'm = Vj(1)',  # Second arg  : j-variable, of size 1 (scalar)
                     'loc = Vj(1)',  # Second arg  : j-variable, of size 1 (scalar)
                     'var = Vj(1)',
                     'p = Pm(1)']

        my_routine1 = Genred(formula1, variables1, reduction_op='LogSumExp', axis=1, dtype=self.dtype)
        result1 = my_routine1(value, self.logits1, self.loc, self.var, self.p, backend=self.device)
        result1 = result1.reshape(shape)
        return torch.where(self._mask, result1, result0)
