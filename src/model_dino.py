"""
DINO, then hidden layers to a conv head.
"""

import torch
import torch.nn as nn


class DinoNN(nn.Module):
    def __init__(self, num_hidden_layers=1):
        """
        num_hidden_layers: Number of hidden layers to use from DINO.
        """
        super().__init__()

        self.num_hidden_layers = num_hidden_layers

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

        self.head = nn.Sequential(
            nn.Conv2d(384 * self.num_hidden_layers, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        """
        x: (B, C, H, W), float 0-1.
        return: (B, 1, H, W), float 0-1.
        """
        with torch.no_grad():
            layers = self.dino.get_intermediate_layers(x, n=self.num_hidden_layers, reshape=True)
            layers = torch.cat(layers, dim=1)

        layers = self.head(layers)
        if not logits:
            layers = self.sigmoid(layers)
        return layers
