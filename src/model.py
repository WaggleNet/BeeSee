import torch
import torch.nn as nn


class ReducedUNet(nn.Module):
    """
    Implementation of smaller U-Net in "Towards dense object tracking in a 2D honeybee hive".

    Input shape: (B, 1, W, H)
    Output shape: (B, 32, W, H)
    """

    def __init__(self):
        super().__init__()

        # Naming: left/right (which side of the U-Net), 1/2/3 (layer)
        self.left1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.left2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.left3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.left4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
        )
        self.right3 = nn.Sequential(
            nn.Conv2d(128+256, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(64+128, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(32+64, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        mid1 = self.left1(x)
        mid2 = self.left2(mid1)
        mid3 = self.left3(mid2)
        mid4 = self.left4(mid3)
        right3 = self.right3(torch.cat((mid3, self.upsample(mid4)), dim=1))
        right2 = self.right2(torch.cat((mid2, self.upsample(right3)), dim=1))
        right1 = self.right1(torch.cat((mid1, self.upsample(right2)), dim=1))
        return right1