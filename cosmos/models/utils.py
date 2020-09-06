import torch


def probs_to_logits(probs):
    return torch.log(probs / (1 - probs))


def pi_m_calc(lamda, S):
    pi_m = torch.eye(S+1)
    pi_m[0] = lamda
    return pi_m


def pi_theta_calc(pi, K, S):
    pi_theta = torch.zeros(S*K+1)
    pi_theta[0] = pi[0]
    for s in range(S):
        for k in range(K):
            pi_theta[K*s + k + 1] = pi[s + 1] / K
    return pi_theta


def theta_trans_calc(A, K, S):
    theta_trans = torch.zeros(K*S+1, K*S+1)
    theta_trans[0, 0] = A[0, 0]
    for s in range(S):
        for k in range(K):
            theta_trans[0, K*s + k + 1] = A[0, s+1] / K
            theta_trans[K*s + k + 1, 0] = A[s+1, 0] / K
            for z in range(S):
                for q in range(K):
                    theta_trans[K*s + k + 1, K*z + q + 1] = A[s+1, z+1] / K
    return theta_trans


def z_probs_calc(theta_probs):
    return theta_probs[..., 1:].sum(dim=-1).cpu().data


def k_probs_calc(m_probs, theta_probs):
    m_probs = (theta_probs.unsqueeze(dim=-1) * m_probs).sum(dim=-2)
    return torch.stack((m_probs[..., 1] + m_probs[..., 3],
                        m_probs[..., 2] + m_probs[..., 3]),
                       dim=-1).squeeze(dim=-2).cpu().data


def theta_probs_calc(m_probs, theta_probs):
    return (m_probs.unsqueeze(dim=-1) * theta_probs[..., 1:]).sum(dim=-2).cpu().data


def j_probs_calc(m_probs, theta_probs):
    return (m_probs.unsqueeze(dim=-1)
            * (torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
            - theta_probs[..., 1:])).sum(dim=-2).cpu().data


def init_calc(pi, lamda):
    init = torch.zeros(8)
    init[0] = pi[0] * lamda[0] * lamda[0]
    init[1] = pi[0] * lamda[0] * lamda[1]
    init[2] = pi[0] * lamda[0] * lamda[1]
    init[3] = pi[0] * lamda[1] * lamda[1]
    init[4] = pi[1] * lamda[0] / 2
    init[5] = pi[1] * lamda[1] / 2
    init[6] = pi[1] * lamda[0] / 2
    init[7] = pi[1] * lamda[1] / 2
    return init


def trans_calc(A, lamda):
    trans = torch.zeros(8, 8)
    trans[0, 0] = A[0, 0] * lamda[0] * lamda[0]
    trans[0, 1] = A[0, 0] * lamda[1] * lamda[0]
    trans[0, 2] = A[0, 0] * lamda[0] * lamda[1]
    trans[0, 3] = A[0, 0] * lamda[1] * lamda[1]
    trans[0, 4] = A[0, 1] * lamda[0] / 2
    trans[0, 5] = A[0, 1] * lamda[1] / 2
    trans[0, 6] = A[0, 1] * lamda[0] / 2
    trans[0, 7] = A[0, 1] * lamda[1] / 2
    trans[1, 0] = A[0, 0] * lamda[0] * lamda[0]
    trans[1, 1] = A[0, 0] * lamda[1] * lamda[0]
    trans[1, 2] = A[0, 0] * lamda[0] * lamda[1]
    trans[1, 3] = A[0, 0] * lamda[1] * lamda[1]
    trans[1, 4] = A[0, 1] * lamda[0] / 2
    trans[1, 5] = A[0, 1] * lamda[1] / 2
    trans[1, 6] = A[0, 1] * lamda[0] / 2
    trans[1, 7] = A[0, 1] * lamda[1] / 2
    trans[2, 0] = A[0, 0] * lamda[0] * lamda[0]
    trans[2, 1] = A[0, 0] * lamda[1] * lamda[0]
    trans[2, 2] = A[0, 0] * lamda[0] * lamda[1]
    trans[2, 3] = A[0, 0] * lamda[1] * lamda[1]
    trans[2, 4] = A[0, 1] * lamda[0] / 2
    trans[2, 5] = A[0, 1] * lamda[1] / 2
    trans[2, 6] = A[0, 1] * lamda[0] / 2
    trans[2, 7] = A[0, 1] * lamda[1] / 2
    trans[3, 0] = A[0, 0] * lamda[0] * lamda[0]
    trans[3, 1] = A[0, 0] * lamda[1] * lamda[0]
    trans[3, 2] = A[0, 0] * lamda[0] * lamda[1]
    trans[3, 3] = A[0, 0] * lamda[1] * lamda[1]
    trans[3, 4] = A[0, 1] * lamda[0] / 2
    trans[3, 5] = A[0, 1] * lamda[1] / 2
    trans[3, 6] = A[0, 1] * lamda[0] / 2
    trans[3, 7] = A[0, 1] * lamda[1] / 2
    trans[4, 0] = A[1, 0] * lamda[0] * lamda[0]
    trans[4, 1] = A[1, 0] * lamda[1] * lamda[0]
    trans[4, 2] = A[1, 0] * lamda[0] * lamda[1]
    trans[4, 3] = A[1, 0] * lamda[1] * lamda[1]
    trans[4, 4] = A[1, 1] * lamda[0] / 2
    trans[4, 5] = A[1, 1] * lamda[1] / 2
    trans[4, 6] = A[1, 1] * lamda[0] / 2
    trans[4, 7] = A[1, 1] * lamda[1] / 2
    trans[5, 0] = A[1, 0] * lamda[0] * lamda[0]
    trans[5, 1] = A[1, 0] * lamda[1] * lamda[0]
    trans[5, 2] = A[1, 0] * lamda[0] * lamda[1]
    trans[5, 3] = A[1, 0] * lamda[1] * lamda[1]
    trans[5, 4] = A[1, 1] * lamda[0] / 2
    trans[5, 5] = A[1, 1] * lamda[1] / 2
    trans[5, 6] = A[1, 1] * lamda[0] / 2
    trans[5, 7] = A[1, 1] * lamda[1] / 2
    trans[6, 0] = A[1, 0] * lamda[0] * lamda[0]
    trans[6, 1] = A[1, 0] * lamda[1] * lamda[0]
    trans[6, 2] = A[1, 0] * lamda[0] * lamda[1]
    trans[6, 3] = A[1, 0] * lamda[1] * lamda[1]
    trans[6, 4] = A[1, 1] * lamda[0] / 2
    trans[6, 5] = A[1, 1] * lamda[1] / 2
    trans[6, 6] = A[1, 1] * lamda[0] / 2
    trans[6, 7] = A[1, 1] * lamda[1] / 2
    trans[7, 0] = A[1, 0] * lamda[0] * lamda[0]
    trans[7, 1] = A[1, 0] * lamda[1] * lamda[0]
    trans[7, 2] = A[1, 0] * lamda[0] * lamda[1]
    trans[7, 3] = A[1, 0] * lamda[1] * lamda[1]
    trans[7, 4] = A[1, 1] * lamda[0] / 2
    trans[7, 5] = A[1, 1] * lamda[1] / 2
    trans[7, 6] = A[1, 1] * lamda[0] / 2
    trans[7, 7] = A[1, 1] * lamda[1] / 2
    return trans