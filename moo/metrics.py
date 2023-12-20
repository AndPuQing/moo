import torch


def crowding_distance(targets: torch.Tensor, fronts: list[torch.Tensor]):
    """
    Compute the crowding distance for a set of solutions.

    targets: shape: (pop_size, n_obj)
    front: shape: (front_size,)

    Returns:
        distances: shape: (front_size,)
    """
    distances = []
    target_min = targets.min(dim=0).values
    target_max = targets.max(dim=0).values
    for i, front in enumerate(fronts):
        front_target = targets[front]
        distance = torch.zeros(len(front), device=front.device)
        _, indices = front_target[:, 0].sort()
        for k in range(1, len(front) - 1):
            for j in range(targets.shape[1]):
                distance[k] += torch.abs(
                    front_target[indices[k + 1], j]
                    - front_target[indices[k - 1], j]
                ) / (target_max[j] - target_min[j])
        distance[0] = torch.inf
        distance[-1] = torch.inf
        distances.append(distance)
    return distances
