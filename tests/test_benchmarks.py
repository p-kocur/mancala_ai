import unittest
import gymnasium as gym
import library.enviroment
import numpy as np
from algorithm.benchmarks import united_policy

class TestBecnhmarks(unittest.TestCase):
    def test_basic(self):
        env = gym.make("Mancala-v0").unwrapped
        s = env.state()[1]
        a = united_policy(s)
        self.assertTrue(isinstance(a, int))