"""
Visualize hidden layers of the model.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import OistDataset
from model import ReducedUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to data dir.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.")
    args = parser.parse_args()

    model = ReducedUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    dataset = OistDataset(args.data, do_aug=False)
    x, _ = random.choice(dataset)
    x = (x.float() / 255).to(DEVICE).unsqueeze(0)
    layers = model.forward(x, hidden_layers=True)

    rows = 3
    cols = len(layers) // rows
    fig, axs = plt.subplots(rows, cols)
    for img_index, (name, layer) in enumerate(layers.items()):
        layer = layer[0].cpu().numpy()

        # 3 axis PCA
        if layer.shape[0] > 3:
            orig_res = layer.shape[1:]
            indices = random.choices(range(orig_res[0] * orig_res[1]), k=256)
            mat = layer.reshape(layer.shape[0], -1)[:, indices].T
            u, s, v = np.linalg.svd(mat)
            comps = v[:, :3]

            img = np.zeros((3, orig_res[0], orig_res[1]))
            for i in range(3):
                img[i] = np.tensordot(layer, comps[:, i], axes=(0, 0))
            img = img.transpose(1, 2, 0).astype(np.float32)
            # Sigmoid
            img = 1 / (1 + np.exp(-img))

        else:
            img = layer[0]

        ax = axs[img_index // cols, img_index % cols]
        ax.set_title(name)
        if img_index == 0:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.axis("off")

    #plt.tight_layout()
    plt.savefig("hidden_layers.jpg")


if __name__ == "__main__":
    main()
