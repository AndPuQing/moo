import torch
from test_base import TestCase

from moo.optim import Bas


class TestBas(TestCase):
    def test_michalewicz(self):
        x = torch.rand(2)
        M = 10
        optimizer = Bas([x])
        for i in range(100):

            def closure(x=x):
                i = torch.arange(1, len(x) + 1)
                temp1 = torch.sin(x) * torch.sin(i * x**2 / torch.pi) ** (
                    2 * M
                )
                return -torch.sum(temp1)

            optimizer.step(closure)
        assert torch.allclose(closure(x), torch.Tensor([-1.8013]), atol=1e-2)
