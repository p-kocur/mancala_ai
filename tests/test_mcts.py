import unittest
import gymnasium as gym
import library.enviroment
import numpy as np
from algorithm.choose_move import selection
import random

class TestSelection(unittest.TestCase):
    def test_basic_1_iteration(self):
        env = gym.make("Mancala-v0").unwrapped
        env.reset()
        def simple_f(s):
            return 1, [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        pi = selection(env.state()[1], simple_f, env, 1)
        self.assertEqual(pi, [0, 0, 0, 0, 0, 0])
        
    def test_basic_2_iteration(self):
        env = gym.make("Mancala-v0").unwrapped
        env.reset()
        def simple_f(s):
            return 1, [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        pi = selection(env.state()[1], simple_f, env, 2)
        self.assertEqual(pi, [1, 0, 0, 0, 0, 0])
        
    def test_basic_3_iteration(self):
        env = gym.make("Mancala-v0").unwrapped
        env.reset()
        def simple_f(s):
            return 1, [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        pi = selection(env.state()[1], simple_f, env, 3)
        self.assertEqual(pi, [1, 1, 0, 0, 0, 0])
        
        
        
if __name__ == "__main__":
    unittest.main()

