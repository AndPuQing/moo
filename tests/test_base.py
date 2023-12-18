import unittest

import torch


class TestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
