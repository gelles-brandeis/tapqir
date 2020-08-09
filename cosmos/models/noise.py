import torch
import torch.distributions.constraints as constraints
from torch.distributions.transforms import AffineTransform
import pyro.distributions as dist

_noise = dict()
_noise_fn = dict()


def CameraUnit2(locs, gain, offset):
    base_distribution = dist.Gamma(locs/gain, 1/gain)
    transforms = [AffineTransform(loc=offset, scale=1)]
    return dist.TransformedDistribution(base_distribution, transforms)


def CameraUnit3(locs, gain, offset, offset_var):
    return dist.Normal(locs + offset, torch.sqrt(locs * gain + offset_var))


def CameraUnit(locs, gain, offset, offset_var):
    shape = (locs + offset) ** 2 / (locs * gain + offset_var)
    rate = shape / (locs + offset)
    return dist.Gamma(shape, rate)


_noise["GammaOffset"] = {
    "offset": {
        # "prior": dist.Uniform(torch.tensor(0.), torch.tensor(90.9)),
        "prior": dist.Uniform(torch.tensor(0.), torch.tensor(1080.)),
        "guide_dist": dist.Delta,
        "guide_params": {
            "v": {"name": "offset_v",
                  "init_tensor": torch.tensor(1000.),
                  "constraint": constraints.interval(0, 1080)}
        }
    },
    "gain": {
        "prior": dist.HalfNormal(torch.tensor(100.)),
        "guide_dist": dist.Delta,
        "guide_params": {
            "v": {"name": "gain_v",
                  "init_tensor": torch.tensor(5.),
                  "constraint": constraints.positive}
        }}
}
_noise_fn["GammaOffset"] = CameraUnit
