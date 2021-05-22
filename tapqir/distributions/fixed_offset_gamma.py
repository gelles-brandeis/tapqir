# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pyro.distributions import Gamma, TransformedDistribution
from torch.distributions.transforms import AffineTransform


class FixedOffsetGamma(TransformedDistribution):
    r"""
    Gamma distribution shifted by :attr:`offset`::

        X ~ Gamma(mean / gain, 1 / gain)
        f(X) = offset + X
        Y = f(X) ~ FixedOffsetGamma(mean, gain, offset)

    :param mean: mean of the Gamma distribution.
    :param gain: scale parameter of the Gamma distribution.
    :param offset: offset parameter.
    """

    has_rsample = True

    def __init__(self, mean, gain, offset, validate_args=None):
        base_dist = Gamma(mean / gain, 1 / gain)
        super(FixedOffsetGamma, self).__init__(
            base_dist, AffineTransform(loc=offset, scale=1), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        """ """
        new = self._get_checked_instance(FixedOffsetGamma, _instance)
        return super(FixedOffsetGamma, self).expand(batch_shape, _instance=new)
