from torch import reshape
from torch import nn
import torch
from torch.nn.modules.activation import ReLU

class DQN_AtariModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=8, stride=4), # 19
            nn.ReLU(),
            nn.Conv2d(32, 64,kernel_size=4, stride=2), #8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), #6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )

    
    def forward(self, x: torch.Tensor):
        """
            Args:
                ``x`` (Tensor): Tensor of batched images.
        """
        x = self.layers(x)
        return x