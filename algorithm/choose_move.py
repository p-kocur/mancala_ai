import numpy as np
import torch
from collections import defaultdict

class Node:
    def __init__(self):
        self.Qs = np.zeros(6)
        self.Ws = np.zeros(6)
        self.Ns = np.zeros(6)
        self.Ps = np.zeros(6)
        self.is_leaf = True
        
def selection(root_s, net, env, n_iter=500):
    # Resetujemy środowisko i przechodzimy do stanu startowego
    env.reset()
    env.set_state(root_s)
    if env.terminated():
        return None
    
    # Inicjalizujemy słownik, w którym zawarte będą wierzchołki drzewa gry
    nodes = defaultdict(Node)
    root_key = tuple(root_s.flatten())

    for _ in range(n_iter):
        # W states przechowujemy ścieżkę 
        env.reset()
        env.set_state(root_s)
        states = [[root_s, None, env.player]]
        # Dopóki nie dotarliśmy do liścia
        current_key = tuple(states[-1][0].flatten())
        while not nodes[current_key].is_leaf:
            # Wybieramy ruch zgodnie ze wzorem
            u_s = np.zeros(6)
            N_sum = 0
            poss_moves = env.possible_moves()
            for i in poss_moves: 
                n = nodes[current_key].Ns[i]
                N_sum += n
                u_s[i] = nodes[current_key].Ps[i] / (1 + n)
            u_s = u_s * np.sqrt(N_sum)
            U_s = u_s + np.array([nodes[current_key].Qs[i] for i in range(6)])
            #n_poss_moves = list(set(range(6)) - set(poss_moves))
            for i in range(6):
                if not i in poss_moves:
                    U_s[i] = -np.inf
            move = np.argmax(U_s)
            env.step(move)
            # i dodajemy kolejny stan do ścieżki
            states[-1][1] = move
            states.append([env.state()[1], None, env.player])
            current_key = tuple(states[-1][0].flatten())
            
            
        # Uzyskujemy wartość stanu i prawdopodobieństwa kolejnych posunięć
        with torch.no_grad():
            p, v = net(states[-1][0])
            v = v.item()
        #env.set_state(states[-1][0])
        
        # Jeśli nie jesteśmy w stanie końcowym
        if not env.terminated():
            # Dokonujemy ekspansji
            nodes[current_key].is_leaf = False
            #player = env.player
            #for i in range(6):
            #    env.set_state(np.copy(states[-1][0]))
            #    env.step(i)
            #    if not tuple(env.state()[1].flatten()) in nodes:
            #        nodes[tuple(env.state()[1].flatten())] = Node() 
            #    env.player = player
                #nodes[states[-1]].nexts.append(nodes[env.state()])
            nodes[current_key].Ps = p.numpy().squeeze()
        else:
            v = env.reward()
            
        if env.player != 0:
            v = -v
         
        # backpropagating - aktualizujemy stany ze ścieżki 
        for node, move, player in states:
            if not move is None: 
                c_node = nodes[tuple(node.flatten())]
                if player == 0:
                    c_node.Ws[move] += v
                else:
                    c_node.Ws[move] -= v
                c_node.Ns[move] += 1
                c_node.Qs[move] = c_node.Ws[move] / c_node.Ns[move]
    
    return nodes[root_key].Ns
        
        
        