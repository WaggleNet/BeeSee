"""
This is meant to be a concise example that is similar to usage in practice.

Test:
- Load model.
- Load data.
- Run thorax detection.
- Extract mean coordinates.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image

from model_dino import DinoNN
from utils import DEVICE, extract_blobs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to bee image.")
    parser.add_argument("--model", type=Path, required=True, help="Path to Dino model file.")
    args = parser.parse_args()

    # Create and load model
    model = DinoNN().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    # Load and process image
    x_img = read_image(str(args.data)).to(DEVICE)
    x_img = x_img.float() / 255.0
    x_img = x_img.mean(dim=0, keepdim=True)   # Grayscale
    x_img = x_img.unsqueeze(0)   # Add batch dimension
    x_img = F.resize(x_img, (448, 448), antialias=True)   # Resize to multiple of 14
    # x_img is (1, 1, H, W) [0, 1]

    # Run model
    pred = model(x_img).squeeze(0).squeeze(0)   # (H / 14, W / 14) [0, 1] prediction
    plt.imshow(pred.cpu().numpy(), cmap="gray")
    plt.savefig("test_thorax.png")

    # Extract mean coordinates using utils
    blobs = extract_blobs(pred, threshold=0.5, blur=0)
    print(f"Found {len(blobs)} blobs.")
    for blob in blobs:
        moment = [torch.sum(x.float()) for x in torch.where(blob)]
        com = [x / blob.sum() for x in moment]   # (Y, X)
        print(f"Blob mean coordinates (Y, X): {com[0].item()}, {com[1].item()}")


if __name__ == "__main__":
    main()
