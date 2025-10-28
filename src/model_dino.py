"""
DINO, then conv head taking in hidden layers.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_fx
from torchvision.transforms import functional as F
from torchvision.io import read_image

from utils import DEVICE

ROOT = Path(__file__).parent


def apply_fx_quant(model):
    # Load calibration data.
    data_path = ROOT.parent / "datasets" / "thorax"
    data = []
    with torch.inference_mode():
        for file in data_path.iterdir():
            if file.suffix == ".jpg" and "img" in file.name:
                img = read_image(str(file)).to(DEVICE).float() / 255.0
                img = F.resize(img, [224, 224])
                img = img.unsqueeze(0)
                data.append(img)

    qconfig = {"": torch.quantization.get_default_qconfig("fbgemm")}
    model = quantize_fx.prepare_fx(model, qconfig, example_inputs=(data[0],))
    with torch.inference_mode():
        for img in data:
            model(img)
    model = quantize_fx.convert_fx(model)

    return model


DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(DEVICE)
#DINO = quantize_model(DINO)
#DINO = apply_fx_quant(DINO, {nn.Linear}, dtype=torch.qint8)
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
