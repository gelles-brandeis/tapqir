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
