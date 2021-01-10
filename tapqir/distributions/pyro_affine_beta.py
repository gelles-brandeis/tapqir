import torch
from pyro.distributions.torch import Beta, TransformedDistribution
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform


class PyroAffineBeta(TransformedDistribution):
    r"""
    Beta distribution scaled by :attr:`scale` and shifted by :attr:`loc`::

        X ~ Beta(concentration1, concentration0)
        f(X) = loc + scale * X
        Y = f(X) ~ PyroAffineBeta(concentration1, concentration0, loc, scale)

    :param float or torch.Tensor concentration1: 1st concentration parameter
        (alpha) for the Beta distribution.
    :param float or torch.Tensor concentration0: 2nd concentration parameter
        (beta) for the Beta distribution.
    :param float or torch.Tensor loc: location parameter.
    :param float or torch.Tensor scale: scale parameter.
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    has_rsample = True

    def __init__(self, concentration1, concentration0, loc, scale, validate_args=None):
        base_dist = Beta(concentration1, concentration0)
        super(PyroAffineBeta, self).__init__(
            base_dist,
            AffineTransform(loc=loc, scale=scale),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PyroAffineBeta, _instance)
        return super(PyroAffineBeta, self).expand(batch_shape, _instance=new)

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample from `Beta` distribution and applies `AffineTransform`.
        Additionally clamps the output in order to avoid `NaN` and `Inf` values
        in the gradients.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            # eps = torch.finfo(x.dtype).eps
            eps = 1e-5
            x = torch.max(torch.min(x, self.low + eps), self.high - eps)
            return x

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample from `Beta` distribution and applies `AffineTransform`.
        Additionally clamps the output in order to avoid `NaN` and `Inf` values
        in the gradients.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        # eps = torch.finfo(x.dtype).eps
        eps = 1e-5
        x = torch.max(torch.min(x, self.low + eps), self.high - eps)
        return x

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)

    @property
    def concentration1(self):
        return self.base_dist.concentration1

    @property
    def concentration0(self):
        return self.base_dist.concentration0

    @property
    def size(self):
        return self.concentration1 + self.concentration0

    @property
    def loc(self):
        return torch.as_tensor(self.transforms[0].loc)

    @property
    def scale(self):
        return torch.as_tensor(self.transforms[0].scale)

    @property
    def low(self):
        return self.loc

    @property
    def high(self):
        return self.loc + self.scale

    @property
    def mean(self):
        return self.loc + self.scale * self.base_dist.mean

    @property
    def variance(self):
        return self.scale.pow(2) * self.base_dist.variance
