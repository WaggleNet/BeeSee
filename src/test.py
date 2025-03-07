import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils import *


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data_type", type=str, choices=["oist"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model_type", type=str, choices=["unet"], required=True)
    args = parser.parse_args()

    dataset_cls, model_cls = get_dataset_model(args.data_type, args.model_type)
    dataset = dataset_cls(args.data)
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    indices = random.sample(range(len(dataset)), 6)

    samples = [dataset[i] for i in indices]
    xs = torch.stack([sample[0] for sample in samples])
    preds = model(xs.to(DEVICE))

    fig, axs = plt.subplots(4, 6, dpi=300)
    axs[0, 0].set_title("X")
    axs[1, 0].set_title("Y")
    axs[2, 0].set_title("Pred")
    axs[3, 0].set_title("Y - Pred")
    for i in range(len(indices)):
        x, y = samples[i]
        x = x.permute(1, 2, 0).cpu().numpy()
        y = y.permute(1, 2, 0).cpu().numpy()
        pred = preds[i]
        pred = pred.permute(1, 2, 0).cpu().numpy()
        axs[0, i].imshow(x)
        axs[1, i].imshow(y)
        axs[2, i].imshow(pred)
        axs[3, i].imshow(y - pred)
        for j in range(4):
            axs[j, i].axis("off")

    #plt.show()
    plt.savefig("test.jpg")


if __name__ == "__main__":
    main()
