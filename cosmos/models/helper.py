import torch
from torch.distributions.transforms import AffineTransform
import pyro
import pyro.distributions as dist

def Location(mode, size, loc, scale):
    """
    Location(mode, size, loc, scale) = loc + scale * Beta((mode - loc) / scale, size)
    mode(Location) = mode
    var(Location) = <Location ** 2> - <Location> ** 2
    <Location> = loc + scale * <Beta>
    <Location> ** 2 = loc ** 2 + (scale ** 2) * (<Beta> ** 2) + 2 * loc * scale * <Beta>
    Location ** 2 = loc ** 2 + (scale ** 2) * (Beta ** 2) + 2 * loc * scale * Beta
    <Location ** 2> = loc ** 2 + (scale ** 2) * <Beta ** 2> + 2 * loc * scale * <Beta>
    var(Location) = (scale ** 2) * (<Beta ** 2> - <Beta> ** 2)
    """ 
    mode = (mode - loc) / scale
    concentration1 = mode * (size - 2) + 1
    concentration0 = (1 - mode) * (size - 2) + 1
    base_distribution = dist.Beta(concentration1, concentration0)
    transforms =  [AffineTransform(loc=loc, scale=scale)]
    return dist.TransformedDistribution(base_distribution, transforms)
