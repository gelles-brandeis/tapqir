import torch
import pyro.distributions as dist
import math


def ScaledBeta(mode, size, loc, scale):
    mode = (mode - loc) / scale
    concentration1 = mode * size
    concentration0 = (1 - mode) * size
    return dist.Beta(concentration1, concentration0)


def bernoulli(x, pi):
    return dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()


def poisson(x, lamda):
    return dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()


def j_param(pi, K):
    """
    p(m) = p(z+j=m) = p(z=1) * p(j=m-1) + p(z=0) * p(j=m)
    """
    j_pi = torch.zeros(2,2)
    j_pi[0] = pi
    j_pi[1] = torch.tensor([1., 0.])
    #j_pi[0,2] = poisson(2, lamda)
    #j_pi[1,0] = poisson(0, lamda)
    #j_pi[1,1] = poisson(1, lamda)
    return j_pi


def m_param(lamda, K):
    """
    p(m) = p(z+j=m) = p(z=1) * p(j=m-1) + p(z=0) * p(j=m)
    """
    m_pi = torch.zeros(2,K+1)
    m_pi[0,0] = poisson(0, lamda)
    m_pi[0,1] = poisson(1, lamda)
    m_pi[0,2] = 1 - m_pi[0].sum()
    m_pi[1,1] = poisson(0, lamda)
    m_pi[1,2] = 1 - m_pi[1].sum()
    return m_pi


def theta_param(pi, lamda, K):
    theta_pi = torch.zeros(2**K, K+1)
    theta_pi[0, 0] = 1
    theta_pi[1, 0] = bernoulli(0, pi) \
        * poisson(1, lamda) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[2, 0] = bernoulli(0, pi) \
        * poisson(1, lamda) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[3, 0] = bernoulli(0, pi) \
        * poisson(2, lamda) / m_param(pi, lamda, K)[3]

    theta_pi[1, 2] = bernoulli(1, pi) \
        * poisson(0, lamda) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[2, 1] = bernoulli(1, pi) \
        * poisson(0, lamda) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[3, 1] = bernoulli(1, pi) \
        * poisson(1, lamda) / (m_param(pi, lamda, K)[3] * 2)
    theta_pi[3, 2] = bernoulli(1, pi) \
        * poisson(1, lamda) / (m_param(pi, lamda, K)[3] * 2)
    return theta_pi
