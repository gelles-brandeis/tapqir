import torch
from pyro.distributions import AffineBeta as PyroAffineBeta
from torch.distributions import constraints


class AffineBeta(PyroAffineBeta):
    r"""
    Beta distribution shifted by :attr:`loc` and scaled by :attr:`scale`::

        concentration1 = size * (mean - low) / (high - low)
        concentration0 = size * (high - mean) / (high - low)
        X ~ Beta(concentration1, concentration0)
        f(X) = low + (high - low) * X
        Y = f(X) ~ AffineBeta(mean, sample_size, low, high)

    :param mean: mean of the distribution.
    :param sample_size: sample size parameter of the Beta distribution.
    :param low: min parameter.
    :param high: max parameter.
    """

    arg_constraints = {
        "mean": constraints.dependent,
        "samle_size": constraints.real,
        "low": constraints.real,
        "high": constraints.dependent,
    }

    def __init__(self, mean, samle_size, low, high, validate_args=None):
        if low != high:
            concentration1 = samle_size * (mean - low) / (high - low)
            concentration0 = samle_size * (high - mean) / (high - low)
        else:
            # this is needed to work with funsor make_dist
            low = torch.tensor(0.0)
            high = torch.tensor(1.0)
            concentration1 = torch.tensor(1.0)
            concentration0 = torch.tensor(1.0)
        super(AffineBeta, self).__init__(
            concentration1,
            concentration0,
            low,
            high - low,
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(AffineBeta, _instance)
        return super(AffineBeta, self).expand(batch_shape, _instance=new)
