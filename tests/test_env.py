import unittest
import gymnasium as gym
import library.enviroment
import numpy as np

class TestEnv(unittest.TestCase):
    def test_basic(self):
        env = gym.make("Mancala-v0").unwrapped
        s = env.state()[1]
        self.assertTrue(np.allclose(s, np.array([[4, 4, 4, 4, 4, 4, 0],
                                                [4, 4, 4, 4, 4, 4, 0]])))
        
    def test_move(self):
        env = gym.make("Mancala-v0").unwrapped
        env.step(5)
        s = env.state()[1]
        print(s)
        self.assertTrue(np.allclose(s, np.array([[5, 5, 5, 4, 4, 4, 0],
                                                 [4, 4, 4, 4, 4, 0, 1]])))
        
    def test_set_state(self):
        env = gym.make("Mancala-v0").unwrapped
        env.set_state(np.array([[4, 0, 0, 4, 0, 0, 10],
                                [4, 0, 0, 4, 0, 0, 10]]))
        s = env.state()[1]
        print(s)
        self.assertTrue(np.allclose(s, np.array([[4, 0, 0, 4, 0, 0, 10],
                                                [4, 0, 0, 4, 0, 0, 10]])))
        
    def test_reward(self):
        env = gym.make("Mancala-v0").unwrapped
        env.set_state(np.array([[4, 0, 0, 4, 0, 0, 5],
                                [4, 0, 0, 4, 0, 0, 25]]))
        _, r, _, _ = env.step(0)
        self.assertEqual(r, 1)
        
if __name__ == "__main__":
    unittest.main()