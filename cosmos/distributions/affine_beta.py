import torch
from torch.distributions.transforms import AffineTransform
from pyro.distributions import Beta, TransformedDistribution


class AffineBeta(TransformedDistribution):
    r"""
    Beta distribution shifted by :attr:`loc` and scaled by :attr:`scale`::

        concentration1 = size * (mean - a) / (b - a)
        concentration0 = size * (b - mean) / (b - a)
        X ~ Beta(concentration1, concentration0)
        f(X) = a + (b - a) * X
        Y = f(X) ~ AffineBeta(mean, size, a, b)

    :param mean: mean of the distribution.
    :param size: size parameter of the Beta distribution.
    :param a: min parameter.
    :param b: max parameter.
    """

    has_rsample = True

    def __init__(self, mean, size, a, b, validate_args=None):
        concentration1 = size * (mean - a) / (b - a)
        concentration0 = size * (b - mean) / (b - a)
        base_dist = Beta(concentration1, concentration0)
        super(AffineBeta, self).__init__(base_dist, AffineTransform(loc=a, scale=(b - a)),
                                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """
        """
        new = self._get_checked_instance(AffineBeta, _instance)
        return super(AffineBeta, self).expand(batch_shape, _instance=new)

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
            x = x.clamp(min=self.loc + eps, max=self.loc + self.scale - eps)
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
        x = x.clamp(min=self.loc + eps, max=self.loc + self.scale - eps)
        return x

    @property
    def loc(self):
        return self.transforms[0].loc

    @property
    def scale(self):
        return self.transforms[0].scale

    @property
    def mean(self):
        return self.loc + self.scale * self.base_dist.mean
