from abc import ABCMeta, abstractmethod

from torch import nn


class Algorithm(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.setUp()

    @abstractmethod
    def setUp(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    @abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError
