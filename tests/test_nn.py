import unittest
import numpy as np
import torch
from torch import nn
import random
import gymnasium as gym
import library.enviroment
import algorithm.nets as nets

class TestNN(unittest.TestCase):
    def test_is_compatible_with_env(self):
        env = gym.make("Mancala-v0").unwrapped
        env.reset()
        net = nets.NeuralNetwork()
        p, v = net(env.state()[1])
        self.assertEqual(p.shape, (1, 6))
        self.assertEqual(v.shape, (1, 1))
        
    def test_train(self):
        net = nets.NeuralNetwork()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        z = 1
        pi = torch.tensor([[0, 1, 1, 0, 0, 1]], dtype=torch.float32)
        
        optimizer.zero_grad()
        p, v = net(np.array([[0, 1, 1, 1, 1, 0, 8],
                             [0, 1, 1, 1, 1, 0, 8]]))
        loss = nets.loss_function(v, p, pi, z)
        loss.backward()
        optimizer.step()
        
    def test_train_matrices(self):
        net = nets.NeuralNetwork()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        z = 1
        x = np.array([[0, 1, 1, 0, 0, 1]])
        pi = torch.tensor(np.array([x, x]), dtype=torch.float32)
        z = torch.tensor([z, z], dtype=torch.float32)
        s = np.array([[0, 1, 1, 1, 1, 0, 8],
                             [0, 1, 1, 1, 1, 0, 8]])
        ss = [s, s]
        
        optimizer.zero_grad()
        p, v = net(ss)
        loss = nets.loss_function(v, p, pi, z)
        loss.backward()
        optimizer.step()
        
    def test_saving_loading(self):
        net = nets.NeuralNetwork()
        torch.save(net, "model/model_test.pth")
        net2 = nets.NeuralNetwork()
        net2 = torch.load("model/model_test.pth", weights_only=False)
        self.assertTrue(str(net2.state_dict()) == str(net.state_dict()))
        
    def test_is_softmax(self):
        env = gym.make("Mancala-v0").unwrapped
        env.reset()
        net = nets.NeuralNetwork()
        p, v = net(env.state()[1])
        self.assertEqual(p.sum(), 1)
    
    
if __name__ == "__main__":
    unittest.main()
        