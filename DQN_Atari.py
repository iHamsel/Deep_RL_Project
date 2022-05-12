from torch import reshape
from torch import nn
import torch
from torch.nn.modules.activation import ReLU

class DQN_AtariModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, 8, kernel_size=3, padding=1, stride=2),   #48x48
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5),                            #44x44
            nn.ReLU(),
            nn.MaxPool2d(2),                                            #22x22
            nn.Conv2d(16, 32, kernel_size=7),                           #16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=2),      #8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    
    def forward(self, x: torch.Tensor):
        """
            Args:
                ``x`` (Tensor): Tensor of batched images.
        """
        x = self.layers(x)
        return x