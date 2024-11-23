import numpy as np
import torch
from collections import defaultdict

class Node:
    def __init__(self):
        self.Qs = [0] * 6
        self.Ws = [0] * 6
        self.Ns = [0] * 6
        self.Ps = []
        self.nexts = []
        self.is_leaf = True
        
d = defaultdict(Node)
print(d[0])