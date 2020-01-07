import torch
import numpy as np
import os
from torch.distributions.transforms import AffineTransform
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from cosmos.models.noise import _noise, _noise_fn
from cosmos.utils.utils import write_summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math


def Location(mean, size, loc, scale):
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
    mean = (mean - loc) / scale
    concentration1 = mean * size
    concentration0 = (1 - mean) * size
    base_distribution = dist.Beta(concentration1, concentration0)
    transforms =  [AffineTransform(loc=loc, scale=scale)]
    return dist.TransformedDistribution(base_distribution, transforms)

def m_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    m_pi = torch.zeros(2**K)
    m_pi[0] = bernoulli(0.) * poisson(0.)
    k = 1
    for m in range(1,K+1):
        r = int(math.factorial(K)/(math.factorial(K-m)*math.factorial(m)))
        for _ in range(r):
            m_pi[k] = (bernoulli(1) * poisson(m-1) + bernoulli(0) * poisson(m)) / r
            k += 1
    return m_pi

def theta_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    theta_pi = torch.zeros(2**K,K+1)
    theta_pi[0,0] = 1
    theta_pi[1,0] = bernoulli(0) * poisson(1) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[1,1] = bernoulli(1) * poisson(0) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[2,0] = bernoulli(0) * poisson(1) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[2,2] = bernoulli(1) * poisson(0) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[3,0] = bernoulli(0) * poisson(2) / m_param(pi, lamda, K)[3]
    theta_pi[3,1] = bernoulli(1) * poisson(1) / (m_param(pi, lamda, K)[3] * 2)
    theta_pi[3,2] = bernoulli(1) * poisson(1) / (m_param(pi, lamda, K)[3] * 2)
    return theta_pi
    #k = 1
    #for m in range(1,K+1):
    #    r = int(math.factorial(K)/(math.factorial(K-m)*math.factorial(m)))
    #    for _ in range(r):
    #        theta_pi[k,0] = bernoulli(0) * poisson(m) / m_param(pi, lamda, K)[m]
    #        theta_pi[k,1:k+1] = bernoulli(1) * poisson(m-1) / m_param(pi, lamda, K)[m]
    #        k += 1
