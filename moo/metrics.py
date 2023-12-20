import torch


def crowding_distance(targets: torch.Tensor, fronts: list[torch.Tensor]):
    """
    Compute the crowding distance for a set of solutions.

    targets: shape: (pop_size, n_obj)
    front: shape: (front_size,)

    Returns:
        distances: shape: (front_size,)
    """
    distances = torch.zeros(len(fronts), device=targets.device)
    for i, front in enumerate(fronts):
        print(f"{i} front: {front}")
