from typing import Callable

import torch
from torch import optim


class Bas(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1,
        eta=0.95,
        c=5,
        eps: float = 1e-8,
    ):
        defaults = {
            "eta": eta,
            "lr": lr,
            "c": c,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["lr"] = group["lr"]
                    state["eta"] = group["eta"]
                    state["c"] = group["c"]
                    state["eps"] = group["eps"]

                lr = state["lr"]
                eta = state["eta"]
                c = state["c"]
                eps = state["eps"]

                d0 = lr / c
                angle = torch.rand(len(p))
                angle = angle / (eps + torch.norm(angle))
                x = p + angle * d0
                fl = closure(x)
                x = p - angle * d0
                fr = closure(x)
                x = p - (lr * angle * torch.sign(fl - fr))

                state["lr"] = lr * eta
                p.data = x
