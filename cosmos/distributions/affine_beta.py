import torch
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform
from pyro.distributions import Beta, TransformedDistribution


class AffineBeta(TransformedDistribution):
    arg_constraints = {
        "size": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive
    }
    has_rsample = True

    def __init__(self, mean, size, loc, scale, validate_args=None):
        mean = (mean - loc) / scale
        concentration1 = mean * size
        concentration0 = (1 - mean) * size
        base_dist = Beta(concentration1, concentration0)
        super(AffineBeta, self).__init__(base_dist, AffineTransform(loc=loc, scale=scale),
                                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(AffineBeta, _instance)
        return super(AffineBeta, self).expand(batch_shape, _instance=new)

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
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
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
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

    @property
    def variance(self):
        return self.scale.pow(2) * self.base_dist.variance
