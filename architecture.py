import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.actor(x)
        return x
    

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.actor(x)
        return x