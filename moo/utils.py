from typing import Union

import numpy as np
import torch


def convert_to_array(x: Union[np.ndarray, list]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise TypeError(f"Expected type: list or np.ndarray. Got: {type(x)}")


class Dominator:
    @staticmethod
    def domination_matrix(values: torch.Tensor) -> torch.Tensor:
        """
        Compute the domination matrix for a set of solutions.

        values: shape: (pop_size, n_obj)

        Returns:
            matrix: shape: (pop_size, pop_size)
            matrix[i, j] = True if solution i dominates solution j
        """
        matrix = (values.unsqueeze(0) <= values.unsqueeze(1)).bool() | (
            values.unsqueeze(0) < values.unsqueeze(1)
        ).bool()
        matrix = matrix.all(dim=2)
        return matrix.fill_diagonal_(False)

    @staticmethod
    def fast_non_dominated_sort(values: torch.Tensor) -> list[torch.Tensor]:
        """
        values: shape: (pop_size, n_obj)

        Returns:
            fronts: list of tensors with shape (front_size,)
        """
        pop_size = values.shape[0]

        # Compute domination matrix
        # shape: (pop_size, pop_size)
        dominance = Dominator.domination_matrix(values)
        fronts = []
        while dominance.shape[0] > 0:
            # Find solutions that are not dominated
            front = (~dominance).all(dim=1)
            fronts.append(torch.arange(pop_size, device=values.device)[front])

            # Remove dominated solutions from the dominance matrix
            dominance = dominance[~front][:, ~front]
            pop_size = dominance.shape[0]

        return fronts


if __name__ == '__main__':
    values = torch.rand(40, 2)
    values = values.to("cuda")
    fronts = Dominator.fast_non_dominated_sort(-values)
