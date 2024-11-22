import numpy as np
import torch

class Node:
    def __init__(self):
        self.Qs = [0] * 6
        self.Ws = [0] * 6
        self.Ns = [0] * 6
        self.Ps = []
        self.nexts = []
        self.is_leaf = True
        
def selection(root_s, net, env, n_iter=50):
    # Resetujemy środowisko i przechodzimy do stanu startowego
    env.reset()
    env.set_state(root_s)
    if env.terminated():
        return None
    
    # Inicjalizujemy słownik, w którym zawarte będą wierzchołki drzewa gry
    nodes = {}
    nodes[str(root_s)] = Node()
    init = np.copy(root_s)

    for _ in range(n_iter):
        # W states przechowujemy ścieżkę 
        env.reset()
        env.set_state(init)
        states = [[np.copy(init), None, env.player]]
        # Dopóki nie dotarliśmy do liścia
        while not nodes[str(states[-1][0])].is_leaf:
            # Wybieramy ruch zgodnie ze wzorem
            u_s = np.zeros(6)
            N_sum = 0
            poss_moves = env.possible_moves()
            for i in poss_moves: 
                n = nodes[str(states[-1][0])].Ns[i]
                N_sum += n
                u_s[i] = nodes[str(states[-1][0])].Ps[i] / (1 + n)
            u_s = u_s * np.sqrt(N_sum)
            U_s = np.full(6, -np.inf)
            U_s = u_s + np.array([nodes[str(states[-1][0])].Qs[i] for i in range(6)])
            #n_poss_moves = list(set(range(6)) - set(poss_moves))
            for i in range(6):
                if not i in poss_moves:
                    U_s[i] = -np.inf
            move = np.argmax(U_s)
            env.step(move)
            # i dodajemy kolejny stan do ścieżki
            states[-1][1] = move
            states.append([np.copy(env.state()[1]), None, env.player])
            
        # Uzyskujemy wartość stanu i prawdopodobieństwa kolejnych posunięć
        with torch.no_grad():
            p, v = net(states[-1][0])
            v = v.item()
        env.set_state(states[-1][0])
        
        # Jeśli nie jesteśmy w stanie końcowym
        if not env.terminated():
            # Dokonujemy ekspansji
            nodes[str(states[-1][0])].is_leaf = False
            player = env.player
            for i in range(6):
                env.set_state(np.copy(states[-1][0]))
                env.step(i)
                if not str(env.state()[1]) in nodes:
                    nodes[str(env.state()[1])] = Node() 
                env.player = player
                #nodes[states[-1]].nexts.append(nodes[env.state()])
            nodes[str(states[-1][0])].Ps = p.numpy().squeeze()
        else:
            v = env.reward()
            
        if env.player != 0:
            v = -v
         
        # backpropagating - aktualizujemy stany ze ścieżki 
        for node, move, player in states:
            if not move is None: 
                if player == 0:
                    nodes[str(node)].Ws[move] += v
                else:
                    nodes[str(node)].Ws[move] -= v
                nodes[str(node)].Ns[move] += 1
                nodes[str(node)].Qs[move] = nodes[str(node)].Ws[move] / nodes[str(node)].Ns[move]
       
    return nodes[str(root_s)].Ns
        
        
        