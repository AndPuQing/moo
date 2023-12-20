import math

import torch

from moo.metrics import crowding_distance


# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


# Function to calculate crowding distance
def raw_crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    distance[0] = math.inf
    distance[len(front) - 1] = math.inf
    for k in range(1, len(front) - 1):
        distance[k] = abs(values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (
            max(values1) - min(values1)
        )
        print(max(values1) - min(values1))
        distance[k] = distance[k] + abs(
            values2[sorted1[k + 1]] - values2[sorted1[k - 1]]
        ) / (max(values2) - min(values2))
    return distance


def convtert_to_python_list(tensor):
    return [tensor[i].item() for i in range(len(tensor))]


def test_crowding_distance():
    targets = torch.rand(10, 2)
    fronts = torch.randperm(3)
    distances = raw_crowding_distance(
        convtert_to_python_list(targets[:, 0]),
        convtert_to_python_list(targets[:, 1]),
        fronts,
    )
    distances2 = crowding_distance(targets, [fronts])
    print(distances)
    print(distances2)
    assert torch.allclose(torch.tensor(distances), distances2[0])
