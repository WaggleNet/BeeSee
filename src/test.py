import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import OistDataset
from model import ReducedUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model", type=Path, help="Model file to use.")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    model = ReducedUNet().to(DEVICE)
    print("Loading from", args.model)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    dataset = OistDataset(args.data)
    indices = random.choices(range(len(dataset)), k=args.samples)
    samples = [dataset[i] for i in indices]

    x = torch.stack([s[0] for s in samples]).to(DEVICE)
    pred = model(x.float() / 255)

    # Make matplotlib plot
    fig, axs = plt.subplots(args.samples, 4)
    axs[0, 0].set_title("Input")
    axs[0, 1].set_title("Prediction")
    axs[0, 2].set_title("Truth")
    axs[0, 3].set_title("Composite")
    for i in range(args.samples):
        img = samples[i][0].squeeze(0).cpu().numpy()
        p = pred[i, 0].detach().cpu().numpy()
        truth = samples[i][1].squeeze(0).cpu().numpy()
        axs[i, 0].imshow(img, cmap="gray")
        axs[i, 1].imshow(p)
        axs[i, 2].imshow(truth)
        composite = np.stack((img, img, img), axis=-1).astype(np.int32)
        composite[:, :, 1] += (p * 255).astype(np.int32)
        composite[:, :, 2] += (truth * 255).astype(np.int32)
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        axs[i, 3].imshow(composite)

    #plt.show()
    plt.savefig("test.png")


if __name__ == "__main__":
    main()
