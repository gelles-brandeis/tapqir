import math

import torch
from pykeops.torch import Genred
from pyro.distributions import Gamma, TorchDistribution


class NormalKDE(TorchDistribution):
    r"""
    Sum of the offset variable and the Gamma distributed variable::
    """

    arg_constraints = {}

    def __init__(self, loc, beta, logits, device="GPU"):
        self.dtype = "float32"
        self.device = device
        self.loc = loc.reshape(-1, 1)
        self.var = (loc / beta).reshape(-1, 1)
        self.logits = logits.reshape(-1, 1)
        if self.device == "GPU":
            self.p = torch.tensor([math.log(math.sqrt(2 * math.pi))]).cuda()
        else:
            self.p = torch.tensor([math.log(math.sqrt(2 * math.pi))])
        # self.dist = Gamma(concentration, rate)
        # batch_shape = loc.shape[:-1]
        batch_shape = torch.Size([])
        event_shape = torch.Size([])
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        shape = value.shape
        value = value.reshape(-1, 1)
        formula = "m-Square(x-loc)/(IntCst(2)*var)-Log(Sqrt(var))-p"
        variables = [
            "x = Vi(1)",  # First arg   : i-variable, of size 1 (scalar)
            "m = Vj(1)",  # Second arg  : j-variable, of size 1 (scalar)
            "loc = Vj(1)",  # Second arg  : j-variable, of size 1 (scalar)
            "var = Vj(1)",
            "p = Pm(1)",
        ]

        my_routine = Genred(
            formula, variables, reduction_op="LogSumExp", axis=1, dtype=self.dtype
        )
        result = my_routine(
            value, self.logits, self.loc, self.var, self.p, backend=self.device
        )
        return result.reshape(shape)


def _log_prob(value, concentration, rate, logits):
    torch.cuda.empty_cache()
    k = concentration.shape[-1]
    print(k)
    # import pdb; pdb.set_trace()
    n = 1000
    if k > n:
        lse1 = _log_prob(value, concentration[..., :n], rate[..., :n], logits[..., :n])
        lse2 = _log_prob(value, concentration[..., n:], rate[..., n:], logits[..., n:])
        result = torch.logsumexp(torch.cat((lse1, lse2), -1), -1, keepdim=True)
    elif k == 1:
        obs_logits = Gamma(concentration, rate).log_prob(value)
        result = obs_logits + logits
    else:
        obs_logits = Gamma(concentration, rate).log_prob(value)
        result = obs_logits + logits
        result = torch.logsumexp(result, -1, keepdim=True)
    print(result.shape)
    return result
