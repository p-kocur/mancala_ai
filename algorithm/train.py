from algorithm.nets import NeuralNetwork, loss_function
from algorithm.choose_move import selection
import torch  
import torch.nn as nn
import random
import numpy as np
import gymnasium as gym
import library.enviroment
import time

def train_loop(n_iterations=100, n_games=50, tau=0.5):
    old_network = NeuralNetwork()
    network = NeuralNetwork()
    old_network.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adagrad(network.parameters())
    
    for i in range(n_iterations):
        print(f"Iteracja: {i}")
        print("Gra:")
        for j in range(n_games):
            print(f"{j}")
            experience = play(network, old_network, j%2)
            
            s, z, pi = experience
            optimizer.zero_grad()
            p, v = network(s)
            loss = loss_function(v, p, pi, z)
            loss.backward()
            optimizer.step()
            
            
            #not_lost = clash(network, old_network)
            #print(not_lost)
            #if not_lost > 60:
            #    old_network.load_state_dict(old_network.state_dict())
            #    break
            old_network.load_state_dict(old_network.state_dict())
            
            
    torch.save(network.state_dict(), "model/model.pth")
    
def return_move(net, env, s, random_v=False):
    Ns = np.array(selection(s, net, env))
    Ns = Ns / Ns.sum()
    if random_v:
        return np.random.choice(range(6), 1, p=Ns).item()
    else:
        return np.argmax(Ns), Ns
             
def play(net1, net2, player):
    env = gym.make("Mancala-v0").unwrapped
    env.reset()
    experience = [[], [], []]
    last_to_play = None
    while not env.terminated():
        if env.player == player:
            s = env.state()[1]
            move, pi = return_move(net1, gym.make("Mancala-v0").unwrapped, s)
            experience[0].append(s)
            experience[2].append(pi)
            _, z, _, _ = env.step(move)
            last_to_play = player
            
        else:
            move, _ = return_move(net2, gym.make("Mancala-v0").unwrapped, env.state()[1])
            _, z, _, _ = env.step(move)
            last_to_play = int(not player)

    if last_to_play == player:
        experience[1] = [z] * len(experience[0])
    else:
        experience[1] = [-z] * len(experience[0])
    return experience[0], torch.tensor(experience[1], dtype=torch.float32), torch.tensor(np.array(experience[2]), dtype=torch.float32)

def clash(net1, net2):
    env = gym.make("Mancala-v0").unwrapped
    env.reset()
    not_lost = 0
    
    for i in range(50):
        env.reset()
        last_to_play = None
        while not env.terminated():
            if env.player == i%2:
                s = env.state()[1]
                move = return_move(net1, gym.make("Mancala-v0").unwrapped, s, random_v=True)
                _, z, _, _ = env.step(move)
                last_to_play = i%2
            else:
                s = env.state()[1]
                move = return_move(net2, gym.make("Mancala-v0").unwrapped, s, random_v=True)
                _, z, _, _ = env.step(move)  
                last_to_play = int(not i%2) 
        
        sign = 1
        if last_to_play != i%2:
            sign = -1
            
        if sign*z >= 0:
            not_lost += 1
            
    return not_lost
        
                