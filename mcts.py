import numpy as np
import gymnasium as gym

nodes = {}

def mcts_train(env, n_episodes, k):
    for _ in range(n_episodes):
        env.reset()
        while not env.terminated():
            s_init = env.state()
            
            for _ in range(k):
                sa_nexts = []
                sa_nexts.append([s_init, None])
                r = 0
                while (not env.terminated()) and (sa_nexts[-1][0] in nodes):
                    a = explore(env, sa_nexts[-1][0])
                    sa_nexts[-1][1] = a
                    new_s, r, _, _ = env.step(a)
                    sa_nexts.append([new_s, None])
                if not sa_nexts[-1][0] in nodes:
                    nodes[sa_nexts[-1][0]] = {"visits": [0]*env.action_space.n, "values": [0]*env.action_space.n}
                if r > 0:
                    for sa in sa_nexts:
                        nodes[sa[0]]["visits"][sa[1]] += 1
                        nodes[sa[0]]["values"][sa[1]] += 1/nodes[sa[0]]["visits"][sa[1]] * (r - nodes[sa[0]]["values"][sa[1]])
                        
def explore(env, s):
    for a_i in nodes[s]["visits"]:
        if a_i == 0:
            return nodes[s]["visits"].index(a_i)
    values = [nodes[s]["values"][a] + np.sqrt(2*np.log(sum(nodes[s]["visits"]))/nodes[s]["visits"][a]) for a in env.action_space]
    return np.argmax(values)

def best(env, s):
    values = [nodes[s]["visits"][a] for a in env.action_space]
    return np.argmax(values)

class Node:
    def __init__(self, state):
        self.state = state
        self.visits = [0] * env.action_space.n
        self.values = [0] * env.action_space.n
    