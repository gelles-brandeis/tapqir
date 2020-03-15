import torch
import pyro.distributions as dist
import math


def ScaledBeta(mode, size, loc, scale):
    mode = (mode - loc) / scale
    concentration1 = mode * size
    concentration0 = (1 - mode) * size
    return dist.Beta(concentration1, concentration0)

def ppi_theta_calc(pi, lamda, K):
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    pi_theta = torch.zeros(2**K,K+1)
    pi_theta[0, 0] = 1
    pi_theta[1, 0] = pi[0] * poisson(1)
    pi_theta[1, 1] = pi[1] * poisson(0)
    pi_theta[2, 0] = pi[0] * poisson(1)
    pi_theta[2, 2] = pi[1] * poisson(0)
    pi_theta[3, 0] = pi[0] * poisson(2)
    pi_theta[3, 1] = pi[1] * poisson(1) / 2
    pi_theta[3, 2] = pi[1] * poisson(1) / 2
    return pi_theta

def A_m_calc(A, lamda, K):
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    A_m = torch.zeros(K+1, 2**K)
    A_m[0, 0] = A[0, 0] * poisson(0)
    A_m[0, 1] = (A[0, 0] * poisson(1) + A[0, 1] * poisson(0)) / 2
    A_m[0, 2] = (A[0, 0] * poisson(1) + A[0, 1] * poisson(0)) / 2
    A_m[0, 3] = A[0, 0] * poisson(2) + A[0, 1] * poisson(1)
    A_m[1, 1] = A[1, 0] * poisson(1) + A[1, 1] * poisson(0)
    A_m[1, 3] = A[1, 0] * poisson(2) + A[1, 1] * poisson(1)
    A_m[2, 2] = A[1, 0] * poisson(1) + A[1, 1] * poisson(0)
    A_m[2, 3] = A[1, 0] * poisson(2) + A[1, 1] * poisson(1)
    return A_m

def A_theta_calc(A, lamda, K):
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    A_theta = torch.zeros(K+1, 2**K, K+1)
    A_theta[:, 0, 0] = 1
    A_theta[0, 1, 0] = A[0, 0] * poisson(1)
    A_theta[1, 1, 0] = A[1, 0] * poisson(1)
    A_theta[2, 1, 0] = A[1, 0] * poisson(1)
    A_theta[0, 1, 1] = A[0, 1] * poisson(0)
    A_theta[1, 1, 1] = A[1, 1] * poisson(0)
    A_theta[2, 1, 1] = A[1, 1] * poisson(0)
    A_theta[0, 2, 0] = A[0, 0] * poisson(1)
    A_theta[1, 2, 0] = A[1, 0] * poisson(1)
    A_theta[2, 2, 0] = A[1, 0] * poisson(1)
    A_theta[0, 2, 2] = A[0, 1] * poisson(0)
    A_theta[1, 2, 2] = A[1, 1] * poisson(0)
    A_theta[2, 2, 2] = A[1, 1] * poisson(0)
    A_theta[0, 3, 0] = A[0, 0] * poisson(2)
    A_theta[1, 3, 0] = A[1, 0] * poisson(2)
    A_theta[2, 3, 0] = A[1, 0] * poisson(2)
    A_theta[0, 3, 1] = A[0, 1] * poisson(1) / 2
    A_theta[1, 3, 1] = A[1, 1] * poisson(1) / 2
    A_theta[2, 3, 1] = A[1, 1] * poisson(1) / 2
    A_theta[0, 3, 2] = A[0, 1] * poisson(1) / 2
    A_theta[1, 3, 2] = A[1, 1] * poisson(1) / 2
    A_theta[2, 3, 2] = A[1, 1] * poisson(1) / 2
    return A_theta

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

def trans_theta_calc(A, K):
    theta_trans = torch.zeros(K+1, K+1)
    theta_trans[0, 0] = A[0, 0]
    theta_trans[0, 1] = A[0, 1] / 2
    theta_trans[0, 2] = A[0, 1] / 2
    theta_trans[1, 0] = A[1, 0]
    theta_trans[1, 1] = A[1, 1] / 2
    theta_trans[1, 2] = A[1, 1] / 2
    theta_trans[2, 0] = A[1, 0]
    theta_trans[2, 1] = A[1, 1] / 2
    theta_trans[2, 2] = A[1, 1] / 2
    return theta_trans

def trans_m_calc(A, lamda, K):
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    m_trans = torch.zeros(2**K, 2**K)
    m_trans[0, 0] = poisson(0) * poisson(0) * A[0, 0]
    m_trans[0, 1] = poisson(0) * poisson(1) * A[0, 0] + poisson(0) * poisson(0) * A[0, 1] / 2
    m_trans[0, 2] = poisson(0) * poisson(1) * A[0, 0] + poisson(0) * poisson(0) * A[0, 1] / 2
    m_trans[0, 3] = poisson(0) * poisson(2) * A[0, 0] + poisson(0) * poisson(1) * A[0, 1] / 2
    m_trans[1, 0] = poisson(1) * poisson(0) * A[0, 0] + poisson(0) * poisson(0) * A[1, 0]
    m_trans[1, 1] = poisson(1) * poisson(1) * A[0, 0] + poisson(0) * poisson(1) * A[1, 0] + poisson(1) * poisson(0) * A[0, 1] / 2
    m_trans[1, 2] = poisson(1) * poisson(1) * A[0, 0] + poisson(0) * poisson(1) * A[1, 0] + poisson(1) * poisson(0) * A[0, 1] / 2
    m_trans[1, 3] = poisson(1) * poisson(2) * A[0, 0] + poisson(0) * poisson(2) * A[1, 0] + poisson(1) * poisson(1) * A[0, 1] / 2 + poisson(0) * poisson(1) * A[1, 1] / 2
    m_trans[2, 0] = poisson(1) * poisson(0) * A[0, 0] + poisson(0) * poisson(0) * A[1, 0]
    m_trans[2, 1] = poisson(1) * poisson(1) * A[0, 0] + poisson(0) * poisson(1) * A[1, 0] + poisson(1) * poisson(0) * A[0, 1] / 2
    m_trans[2, 2] = poisson(1) * poisson(1) * A[0, 0] + poisson(0) * poisson(1) * A[1, 0] + poisson(1) * poisson(0) * A[0, 1] / 2
    m_trans[2, 3] = poisson(1) * poisson(2) * A[0, 0] + poisson(0) * poisson(2) * A[1, 0] + poisson(1) * poisson(1) * A[0, 1] / 2 + poisson(0) * poisson(1) * A[1, 1] / 2
    m_trans[3, 0] = poisson(2) * poisson(1) * A[0, 0] + poisson(1) * poisson(1) * A[1, 0]
    m_trans[3, 1] = poisson(2) * poisson(1) * A[0, 0] + poisson(1) * poisson(1) * A[1, 0] + poisson(2) * poisson(0) * A[0, 1] / 2 + poisson(1) * poisson(0) * A[1, 1] / 2
    m_trans[3, 2] = poisson(2) * poisson(1) * A[0, 0] + poisson(1) * poisson(1) * A[1, 0] + poisson(2) * poisson(0) * A[0, 1] / 2 + poisson(1) * poisson(0) * A[1, 1] / 2
    m_trans[3, 3] = poisson(2) * poisson(2) * A[0, 0] + poisson(1) * poisson(2) * A[1, 0] + poisson(2) * poisson(1) * A[0, 1] / 2 + poisson(1) * poisson(1) * A[1, 1] / 2
    return m_trans

def pi_theta_calc(pi, K):
    pi_theta = torch.zeros(K+1)
    pi_theta[0] = pi[0]
    pi_theta[1] = pi[1] / 2
    pi_theta[2] = pi[1] / 2
    return pi_theta

def pi_m_calc(pi, lamda, K):
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    pi_m = torch.zeros(2**K)
    pi_m[0] = pi[0] * poisson(0)
    pi_m[1] = (pi[1] * poisson(0) + pi[0] * poisson(1)) / 2
    pi_m[2] = (pi[1] * poisson(0) + pi[0] * poisson(1)) / 2
    pi_m[3] = pi[1] * poisson(1) + pi[0] * poisson(2)
    return pi_m
