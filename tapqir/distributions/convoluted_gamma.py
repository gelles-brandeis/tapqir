import torch
from torch.distributions import constraints
from pyro.distributions import TorchDistribution, Gamma, Categorical


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

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
        "samples": constraints.real_vector,
        "log_weights": constraints.real_vector
    }
    support = constraints.positive

    def __init__(self, concentration, rate, samples, log_weights, validate_args=None):
        self.concentration = concentration
        self.rate = rate
        self.samples = samples
        self.log_weights = log_weights
        if isinstance(concentration, torch.Tensor):
            concentration = concentration.unsqueeze(-1)
        self.dist = Gamma(concentration, rate)
        self.samples = samples
        self.log_weights = log_weights
        batch_shape = self.dist.batch_shape[:-1]
        event_shape = self.dist.event_shape
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        odx = Categorical(logits=self.log_weights).expand(self.batch_shape).sample()
        offset = self.samples[odx]
        signal = self.dist.sample().squeeze(-1)
        return signal + offset

    def log_prob(self, value):
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(-1)
        mask = value > self.samples
        value = torch.where(mask, value - self.samples, value.new_ones(()))

        obs_logits = self.dist.log_prob(value)
        result = obs_logits + self.log_weights
        result = result.masked_fill(~mask, -40.)
        result = torch.logsumexp(result, -1)
        return result
