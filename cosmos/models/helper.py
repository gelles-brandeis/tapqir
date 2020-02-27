import torch
import pyro.distributions as dist
import math


def ScaledBeta(mode, size, loc, scale):
    mode = (mode - loc) / scale
    concentration1 = mode * size
    concentration0 = (1 - mode) * size
    return dist.Beta(concentration1, concentration0)
"""
def bernoulli(x, pi):
    return dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()


def poisson(x, lamda):
    return dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()

def m_param(lamda, K):
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
"""

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

def z_probs_calc(m_probs, theta_probs):
    return (m_probs * theta_probs[..., 1:].sum(dim=-1)).sum(dim=-1).cpu().data

def k_probs_calc(m_probs):
    return torch.stack((m_probs[..., 1] + m_probs[..., 3],
                        m_probs[..., 2] + m_probs[..., 3]),
                        dim=-1).squeeze(dim=-2).cpu().data

def theta_probs_calc(m_probs, theta_probs):
    return (m_probs.unsqueeze(dim=-1) * theta_probs[..., 1:]).sum(dim=-2).cpu().data

def j_probs_calc(m_probs, theta_probs):
    return (m_probs.unsqueeze(dim=-1) * (torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) - theta_probs[..., 1:])).sum(dim=-2).cpu().data
