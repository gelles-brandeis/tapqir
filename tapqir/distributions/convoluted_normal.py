import torch
from pyro.distributions import Normal, TorchDistribution


class ConvolutedNormal(TorchDistribution):
    r"""
    Sum of the offset variable and the Normal distributed variable::

        X ~ P(samples, log_weights)
        Y ~ Normal(loc, scale)
        Z = X + Y ~ OffsetedNormal(loc, scale, samples, log_weights)

    :meth:`log_prob` is calculated as the convolution of the offset probability
    and the Normal distribution:

        :math:`p(X) = \sum_i p(\text{offset}_i) \text{Normal}(X - \text{offset}_i)`

    :param loc: loc parameter of the Normal distribution.
    :param scale: scale parameter of the Normal distribution.
    :param ~torch.Tensor samples: offset samples.
    :param ~torch.Tensor log_weights: log weights corresponding to the offset samples.
    """

    arg_constraints = {}  # nothing to be constrained

    def __init__(self, loc, scale, samples, log_weights):
        self.dist = Normal(loc.unsqueeze(-1), scale.unsqueeze(-1))
        self.samples = samples
        self.log_weights = log_weights
        batch_shape = self.dist.batch_shape[:-1]
        event_shape = self.dist.event_shape
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        value = value.unsqueeze(-1)
        value = value - self.samples

        obs_logits = self.dist.log_prob(value)
        result = obs_logits + self.log_weights
        result = torch.logsumexp(result, -1)
        return result
