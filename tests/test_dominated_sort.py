import torch

from moo.utils import Dominator


def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (
                (values1[p] > values1[q] and values2[p] > values2[q])
                or (values1[p] >= values1[q] and values2[p] > values2[q])
                or (values1[p] > values1[q] and values2[p] >= values2[q])
            ):
                if q not in S[p]:
                    S[p].append(q)
            elif (
                (values1[q] > values1[p] and values2[q] > values2[p])
                or (values1[q] >= values1[p] and values2[q] > values2[p])
                or (values1[q] > values1[p] and values2[q] >= values2[p])
            ):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


def test_fast_non_dominated_sort():
    values = torch.rand(100, 2)
    fronts1 = Dominator.fast_non_dominated_sort(values)
    fronts2 = fast_non_dominated_sort(values[:, 0], values[:, 1])
    for i in range(len(fronts1)):
        a = set(fronts1[i].cpu().numpy())
        b = set(fronts2[i])
        assert a == b
