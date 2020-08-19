import torch
from pyro.distributions import TorchDistribution, Normal


class ConvolutedNormal(TorchDistribution):
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
