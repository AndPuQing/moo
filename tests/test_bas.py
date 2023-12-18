import torch

from moo.optim import Bas


def test_michalewicz():
    torch.manual_seed(0)
    x = torch.rand(2)
    M = 2
    optimizer = Bas([x])
    for i in range(100):

        def closure(x=x):
            i = torch.arange(1, len(x) + 1)
            temp1 = torch.sin(x) * torch.sin(i * x**2 / torch.pi) ** (2 * M)
            return -torch.sum(temp1)

        optimizer.step(closure)
    assert torch.allclose(closure(x), torch.Tensor([-1.8013]), atol=1e-1)
