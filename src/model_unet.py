import torch
import torch.nn as nn

# Base channels in conv layers.
CHANNELS = 32


class ReducedUNet(nn.Module):
    """
    Implementation of smaller U-Net in "Towards dense object tracking in a 2D honeybee hive".

    Input shape: (B, 1, W, H)
    Output shape: (B, 1, W, H)
    """

    def __init__(self, init_weights=True):
        super().__init__()

        CH = CHANNELS

        # Naming: left/right (which side of the U-Net), 1/2/3 (layer)
        self.left1 = nn.Sequential(
            nn.Conv2d(1, CH, 3, padding=1),
            nn.BatchNorm2d(CH),
            nn.LeakyReLU(),

            nn.Conv2d(CH, CH, 3, padding=1),
            nn.BatchNorm2d(CH),
            nn.LeakyReLU(),
        )
        self.left1_res = nn.Conv2d(1, CH, 1, padding=0)
        self.left2 = nn.Sequential(
            nn.Conv2d(CH, 2*CH, 3, padding=1),
            nn.BatchNorm2d(2*CH),
            nn.LeakyReLU(),

            nn.Conv2d(2*CH, 2*CH, 3, padding=1),
            nn.BatchNorm2d(2*CH),
            nn.LeakyReLU(),
        )
        self.left2_res = nn.Conv2d(CH, 2*CH, 1, padding=0)
        self.left3 = nn.Sequential(
            nn.Conv2d(2*CH, 4*CH, 3, padding=1),
            nn.BatchNorm2d(4*CH),
            nn.LeakyReLU(),

            nn.Conv2d(4*CH, 4*CH, 3, padding=1),
            nn.BatchNorm2d(4*CH),
            nn.LeakyReLU(),
        )
        self.left3_res = nn.Conv2d(2*CH, 4*CH, 1, padding=0)
        self.left4 = nn.Sequential(
            nn.Conv2d(4*CH, 8*CH, 3, padding=1),
            nn.BatchNorm2d(8*CH),
            nn.LeakyReLU(),

            nn.Conv2d(8*CH, 8*CH, 3, padding=1),
            nn.BatchNorm2d(8*CH),
            nn.LeakyReLU(),
        )
        self.left4_res = nn.Conv2d(4*CH, 8*CH, 1, padding=0)
        self.right3 = nn.Sequential(
            nn.Conv2d(12*CH, 4*CH, 3, padding=1),
            nn.BatchNorm2d(4*CH),
            nn.LeakyReLU(),

            nn.Conv2d(4*CH, 4*CH, 3, padding=1),
            nn.BatchNorm2d(4*CH),
            nn.LeakyReLU(),
        )
        self.right3_res = nn.Conv2d(12*CH, 4*CH, 1, padding=0)
        self.right2 = nn.Sequential(
            nn.Conv2d(6*CH, 2*CH, 3, padding=1),
            nn.BatchNorm2d(2*CH),
            nn.LeakyReLU(),

            nn.Conv2d(2*CH, 2*CH, 3, padding=1),
            nn.BatchNorm2d(2*CH),
            nn.LeakyReLU(),
        )
        self.right2_res = nn.Conv2d(6*CH, 2*CH, 1, padding=0)
        self.right1 = nn.Sequential(
            nn.Conv2d(3*CH, CH, 3, padding=1),
            nn.BatchNorm2d(CH),
            nn.LeakyReLU(),

            nn.Conv2d(CH, CH, 3, padding=1),
            nn.BatchNorm2d(CH),
            nn.LeakyReLU(),
        )
        self.right1_res = nn.Conv2d(3*CH, CH, 1, padding=0)

        self.head = nn.Sequential(
            nn.Conv2d(CH, 1, 1),
        )

        self.pooling = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self.init_weights()

    def forward(self, x, logits=False, hidden_layers=False):
        mid1 = self.left1(x) + self.left1_res(x)
        mid2 = self.left2(self.pooling(mid1)) + self.left2_res(self.pooling(mid1))
        mid3 = self.left3(self.pooling(mid2)) + self.left3_res(self.pooling(mid2))
        mid4 = self.left4(self.pooling(mid3)) + self.left4_res(self.pooling(mid3))
        in3 = torch.cat((mid3, self.upsample(mid4)), dim=1)
        right3 = self.right3(in3) + self.right3_res(in3)
        in2 = torch.cat((mid2, self.upsample(right3)), dim=1)
        right2 = self.right2(in2) + self.right2_res(in2)
        in1 = torch.cat((mid1, self.upsample(right2)), dim=1)
        right1 = self.right1(in1) + self.right1_res(in1)
        y = self.head(right1)
        if not logits:
            y = self.sigmoid(y)

        if hidden_layers:
            return {
                "input": x,
                "mid1": mid1,
                "mid2": mid2,
                "mid3": mid3,
                "mid4": mid4,
                "right3": right3,
                "right2": right2,
                "right1": right1,
                "output": y,
            }
        else:
            return y

    def init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

        self.apply(init_fn)
