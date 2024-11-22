import library.enviroment
import gymnasium as gym
import torch
from algorithm.nets import NeuralNetwork
from algorithm.train import return_move

def play():
    env = gym.make("Mancala-v0").unwrapped
    env.reset()
    net = NeuralNetwork()
    net.load_state_dict(torch.load("model/model.pth", weights_only=True))
    last_to_play = None
    
    while not env.terminated():
        print(env._second_player_holes[::-1])
        print(env._first_player_holes)
        print("\n\n")
        if env.player == 0:
            move = int(input("choose your move: ")) - 1
            _, z, _, _ = env.step(move)
            last_to_play = 0
        else:
            move, _ = return_move(net, gym.make("Mancala-v0").unwrapped, env.state()[1])
            _, z, _, _ = env.step(move)
            last_to_play = 1
         
    if last_to_play == 1:
        z *= -1
    if z == 0:
        print("Remis")
    if z == 1:
        print("Wygrana gracza")
    else:
        print("Wygrana modelu")
    
    