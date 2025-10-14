"""
DINO, then conv head taking in hidden layers.
"""

import torch
import torch.nn as nn

from utils import DEVICE

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(DEVICE)
DINO.eval()


class DinoNN(nn.Module):
    def __init__(self, num_hidden_layers=2):
        """
        num_hidden_layers: Number of hidden layers to use from DINO.
        """
        super().__init__()

        self.num_hidden_layers = num_hidden_layers

        self.head = nn.Conv2d(384 * self.num_hidden_layers, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        """
        x: (B, 1, H, W), float 0-1.
        return: (B, 1, H, W), float 0-1.
        """
        with torch.no_grad():
            x = torch.cat([x, x, x], dim=1)
            x = DINO.get_intermediate_layers(x, n=self.num_hidden_layers, reshape=True)
            x = torch.cat(x, dim=1)

        x = self.head(x)
        if not logits:
            x = self.sigmoid(x)
        return x
