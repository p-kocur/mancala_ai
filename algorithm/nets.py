import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as TF

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 1, kernel_size=(2, 1)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(6, 128)
        )
        self.simple_nn = nn.Linear(2, 2)
        self.classifier_p = nn.Sequential(
            nn.Linear(128 + 2, 6),
            nn.Softmax(dim=1))
        self.classifier_v = nn.Linear(128 + 2, 1)
        
    def forward(self, s):
        scores, state = state_to_tensor(s)
        x_1 = self.cnn(state)
        x_2 = self.simple_nn(scores)
        x = torch.cat((x_1, x_2), dim=1)
        return self.classifier_p(x), self.classifier_v(x)
            
def state_to_tensor(ss):
    scores = []
    states = []
    if not isinstance(ss, list):
        ss = [ss]
    for s in ss:
        tensor_row_1 = np.copy(s[1, :-1])
        tensor_row_1 = np.flip(tensor_row_1)
        tensor_row_2 = np.copy(s[0, :-1])
        new_array = np.array([[tensor_row_1, tensor_row_2]])
        scores.append([s[1, -1], s[0, -1]])
        states.append(new_array)
    return torch.tensor(np.array(scores), dtype=torch.float32), torch.tensor(np.array(states), dtype=torch.float32)

def loss_function(v, p, pi, z):
    loss = torch.sum((v-z)**2 - torch.tensordot(pi, torch.log(p)))
    return loss
    