import torch
from torch.distributions import constraints
from pyro.distributions import TorchDistribution, Gamma


class ConvolutedGamma(TorchDistribution):
    r"""
    Sum of the offset variable and the Gamma distributed variable::

        X ~ P(samples, log_weights)
        Y ~ Gamma(concentration, rate)
        Z = X + Y ~ OffsetedGamma(convolution, rate, samples, log_weights)

    :meth:`log_prob` is calculated as the convolution of the offset probability
    and the Gamma distribution:

        :math:`p(X) = \sum_i p(\text{offset}_i) \text{Gamma}(X - \text{offset}_i)`

    :param concentration: shape parameter (alpha) of the Gamma distribution.
    :param rate: rate parameter (beta) of the Gamma distribution.
    :param ~torch.Tensor samples: offset samples.
    :param ~torch.Tensor log_weights: log weights corresponding to the offset samples.
    """

    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}

    def __init__(self, concentration, rate, samples, log_weights):
        self.dist = Gamma(concentration.unsqueeze(-1), rate)
        self.samples = samples
        self.log_weights = log_weights
        batch_shape = self.dist.batch_shape[:-1]
        event_shape = self.dist.event_shape
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        value = value.unsqueeze(-1)
        mask = value > self.samples
        value = torch.where(mask, value - self.samples, value.new_ones(()))

        obs_logits = self.dist.log_prob(value)
        result = obs_logits + self.log_weights
        result = result.masked_fill(~mask, -40.)
        result = torch.logsumexp(result, -1)
        return result
