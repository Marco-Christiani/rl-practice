import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, obs, actions):
        super().__init__()
        self.fc1 = nn.Linear(obs, obs)
        self.fc2 = nn.Linear(obs, actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(torch.tanh(x))
        return x
