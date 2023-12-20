from typing import Callable, Optional, Union

import heartrate
import numpy as np
import torch

from moo.algorithm.base import Algorithm
from moo.metrics import crowding_distance
from moo.utils import Dominator, convert_to_array

heartrate.trace()


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
        self._generate_offspring()
        # P+Q
        self.population = torch.cat([self.population, self.offspring])
        targets = self._evaluate(self.population)  # shape: (2*pop_size, n_obj)
        fronts = Dominator.fast_non_dominated_sort(targets)
        # Compute crowding distance for each front
        distances = crowding_distance(targets, fronts)
        new_solution = []
        for i in range(0, len(fronts)):
            front = fronts[i][torch.argsort(distances[i], descending=True)]
            # front22 = torch.sort(distances[i]).indices
            # front = [fronts[i][front22[j]] for j in range(0, len(fronts[i]))]
            # front.reverse()

            new_solution.extend(front[: self.pop_size - len(new_solution)])
            if len(new_solution) == self.pop_size:
                break
            # for value in front:
            #     new_solution.append(value)
            #     if len(new_solution) == self.pop_size:
            #         break
            # if len(new_solution) == self.pop_size:
            #     break
        new_solution = torch.tensor(new_solution, device=self.device)
        self.population = self.population[new_solution]

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

    def forward(self):
        return self._evaluate(self.population)

    def _crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """
        parents: shape: (2, pop_size)
        """
        random_indices = torch.rand(parents.shape[1], device=self.device) < 0.5
        offspring = torch.where(
            random_indices,
            self._mutation(parents[0] + parents[1]) / 2,
            self._mutation(parents[0] - parents[1]) / 2,
        )
        return offspring

    def _mutation(self, parents: torch.Tensor) -> torch.Tensor:
        """
        target: shape: (pop_size, n_obj)
        """
        mutation_prob = torch.rand(parents.shape, device=self.device) < 1
        mutation = (
            torch.rand(parents.shape, device=self.device)
            * (self.bounds[:, 1] - self.bounds[:, 0])
            + self.bounds[:, 0]
        )
        return torch.where(mutation_prob, mutation, parents)

    def _generate_offspring(self):
        random_indices = torch.randint(
            0, self.pop_size, (2, self.pop_size), device=self.device
        )
        parents = self.population[random_indices]  # shape: (2, pop_size)
        offspring = self._crossover(parents)  # shape: (pop_size)
        self.offspring = offspring


if __name__ == '__main__':

    def fn1(x: torch.Tensor) -> torch.Tensor:
        return -(x**2)

    def fn2(x: torch.Tensor) -> torch.Tensor:
        return -((x - 2) ** 2)

    max_gen = 921

    nsga = NSGAII(target_fn=[fn1, fn2], bounds=[(-55, 55)], pop_size=20)

    gen_no = 0
    while gen_no < max_gen:
        nsga.step()
        gen_no += 1

    import matplotlib
    from matplotlib import pyplot as plt

    matplotlib.use("Agg")
    target = nsga.forward().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(-target[:, 0], -target[:, 1])
    fig.savefig("nsga.png")
