import torch
import torch.nn as nn

class RadarMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 128),   
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024)  
        )

    def forward(self, x):
        return self.mlp(x)
