import torch


def mse_loss(pred, p):
    # print(pred)
    alpha = pred + 1
    S = torch.sum(alpha, axis=1, keepdim=True)
    # print(S.size())
    # print(alpha)
    # print(p)
    E = alpha - 1
    m = alpha / S

    A = torch.sum((p - m) ** 2, axis=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdim=True)
    # if torch.mean(A+B) is np.nan:
    # print(alpha)
    return torch.mean(A + B)