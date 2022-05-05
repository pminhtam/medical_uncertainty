import torch


def calc_uncertainly(pred,num_class=16):
  alpha = pred + 1
  S = torch.sum(alpha, axis=1,keepdim=True)
  return num_class/S