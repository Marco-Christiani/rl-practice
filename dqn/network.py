import torch
from torch import nn

#
# class Network(nn.Module):
#     def __init__(self, obs, actions):
#         super().__init__()
#         self.fc1 = nn.Linear(obs, obs)
#         self.fc2 = nn.Linear(obs, actions)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(torch.tanh(x))
#         return x


def get_layer(dim0, dim1):
    return nn.Sequential(
        nn.Linear(dim0, dim1),
        nn.ReLU()
    )


def build_network(dims):
    return [get_layer(dims[i], dims[i+1]) for i in range(len(dims)-1)]


class Network(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128, n_hidden=2):
        super().__init__()
        self.network = nn.Sequential(
            *build_network([input_dim] + [hidden_dim] * n_hidden))
        self.q_values = nn.Linear(hidden_dim, n_actions)

    def forward(self, X):
        X = self.network(X)
        q_values = self.q_values(X)
        return q_values
