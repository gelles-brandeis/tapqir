import os
import math
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.ops.indexing import Vindex


def m_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(float(x)).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(float(x)).exp()
    m_pi = torch.zeros(2**K)
    m_pi[0] = bernoulli(0.) * poisson(0.)
    k = 1
    for m in range(1,K+1):
        r = int(math.factorial(K)/(math.factorial(K-m)*math.factorial(m)))
        for _ in range(r):
            m_pi[k] = (bernoulli(1) * poisson(m-1) + bernoulli(0) * poisson(float(m))) / r
            k += 1
    return m_pi

def theta_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(float(x)).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(float(x)).exp()
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

print(m_param(torch.tensor([0.3, 0.7]), torch.tensor([0.2]), 2))
print(theta_param(torch.tensor([0.3, 0.7]), torch.tensor([0.2]), 2))
