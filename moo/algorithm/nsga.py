from typing import Callable, Optional, Union

import numpy as np
import torch

from moo.algorithm.base import Algorithm
from moo.metrics import crowding_distance
from moo.utils import Dominator, convert_to_array


class NSGAII(Algorithm):
    def __init__(
        self,
        target_fn: Union[list[Callable], Callable],
        bounds: Union[list[tuple[float, float]], tuple[float, float]],
        pop_size: int,
        device: Optional[torch.device] = None,
    ):
        bounds = convert_to_array(bounds)
        if bounds.ndim != 2:
            raise ValueError(f"Expected bounds to be 2D. Got: {bounds.ndim}")
        if bounds.shape[1] != 2:
            raise ValueError(
                f"Expected bounds to have shape (n, 2). Got: {bounds.shape}"
            )
        if not np.all(bounds[:, 0] <= bounds[:, 1]):
            raise ValueError(
                f"Expected bounds[:, 0] <= bounds[:, 1]. Got: {bounds}"
            )
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.bounds = torch.tensor(bounds, device=self.device)
        self.pop_size = pop_size
        self.target_fn = (
            target_fn if isinstance(target_fn, list) else [target_fn]
        )

        super().__init__()

    def setUp(self):
        self.population = self._init_population()

    def step(self):
        targets = self._evaluate(self.population)
        fronts = Dominator.fast_non_dominated_sort(targets)
        # Compute crowding distance for each front
        distances = crowding_distance(targets, fronts)

    def _init_population(self):
        return (
            torch.rand(self.pop_size, device=self.device)
            * (self.bounds[:, 1] - self.bounds[:, 0])
            + self.bounds[:, 0]
        ).float()

    def _evaluate(self, population: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [fn(population) for fn in self.target_fn]
        ).T  # shape: (pop_size, n_obj)


if __name__ == '__main__':

    def fn1(x: torch.Tensor) -> torch.Tensor:
        return -(x**2)

    def fn2(x: torch.Tensor) -> torch.Tensor:
        return -((x - 2) ** 2)

    max_gen = 3

    nsga = NSGAII(target_fn=[fn1, fn2], bounds=[(-55, 55)], pop_size=20)

    gen_no = 0
    while gen_no < max_gen:
        nsga.step()
        gen_no += 1
