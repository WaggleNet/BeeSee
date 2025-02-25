import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image

from constants import *
from dataset import ThoraxDataset
from model import DinoSegmentation


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the model file.")
    args = parser.parse_args()

    model = DinoSegmentation().to(DEVICE)
    model.load_state_dict(torch.load(str(args.model), map_location=DEVICE))

    if args.data.is_dir():
        dataset = ThoraxDataset(args.data)

        xs = []
        ys = []
        indices = random.choices(range(len(dataset)), k=6)
        for i in range(6):
            x, y = dataset[indices[i]]
            xs.append(x)
            ys.append(y)
        xs = torch.stack(xs).to(DEVICE)
        ys = torch.stack(ys).to(DEVICE)

    else:
        xs = read_image(str(args.data)).float() / 255
        xs = xs.unsqueeze(0).to(DEVICE)
        xs = F.resize(xs, (448, 448))
        ys = None

    preds = model(xs)

    # shape (B, H, W, C)
    xs = (xs.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    if ys is not None:
        ys = ys.squeeze(1).cpu().numpy()
    preds = preds.squeeze(1).cpu().numpy()

    x_res = xs.shape[1:3]

    # Plot three rows: Xs, Preds on X, Ys on X.
    fig, axs = plt.subplots(3, max(xs.shape[0], 2), figsize=(18, 9))
    axs[0, 0].set_title("X")
    axs[1, 0].set_title("Pred over X")
    axs[2, 0].set_title("Y over X")
    for i in range(xs.shape[0]):
        axs[0, i].imshow(xs[i])

        pred_img = xs[i].copy()
        pred_img[cv2.resize(preds[i], x_res) > 0.5, :] = (0, 255, 0)
        axs[1, i].imshow(pred_img)

        if ys is not None:
            y_img = xs[i].copy()
            y_img[cv2.resize(ys[i], x_res) > 0.5, :] = (0, 255, 0)
            axs[2, i].imshow(y_img)

    plt.tight_layout()
    # plt.show()
    plt.savefig("test.png")


if __name__ == "__main__":
    main()
