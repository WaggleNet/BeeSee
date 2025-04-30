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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from model_dino import DinoNN
from utils import DEVICE, extract_blobs, load_dino_model, preprocess_images


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to bee image.")
    parser.add_argument("--model", type=Path, required=True, help="Path to Dino model file.")
    args = parser.parse_args()

    # Create and load model
    model = load_dino_model(args.model)

    # Load and process image
    x_img = cv2.imread(str(args.data))
    x_img = preprocess_images(x_img)[0]
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
