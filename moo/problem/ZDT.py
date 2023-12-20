import torch


def ZDT1(x):
    def g(x):
        return 1 + 9 * (x.sum(dim=1) - x[:, 0]) / (x.shape[1] - 1)

    f1 = x[:, 0]
    f2 = g(x) * (1 - (f1 / g(x)) ** 0.5)
    return torch.stack([f1, f2], dim=1)
