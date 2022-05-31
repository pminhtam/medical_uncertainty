import torch


def KL(alpha):
    beta = torch.ones((1, alpha.shape[1]), device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def mse_loss(pred, p):
    # print(pred)
    alpha = pred + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    # print(S.size())
    # print(alpha)
    # print(p)
    E = alpha - 1
    m = alpha / S

    A = torch.sum((p - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    # if torch.mean(A+B) is np.nan:
    # print(alpha)
    return torch.mean(A + B)