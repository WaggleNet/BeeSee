import argparse
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import WDDDataset


def show_sample(imgs):
    for img in imgs:
        plt.clf()
        plt.imshow(img)
        plt.pause(0.1)


def plot_changing_dft(imgs):
    """
    Each time series is single pixel over time, i.e. imgs[:, i, j]

    imgs: (T, H, W)
    """
    imgs = imgs.transpose(1, 2, 0)  # (H, W, T)
    imgs = imgs * 2 - 1

    freq = 2 * math.pi / imgs.shape[2]
    x = np.arange(imgs.shape[2])
    while freq <= math.pi:
        magnitude = np.hypot(
            (imgs * np.cos(freq * x)).mean(axis=2),
            (imgs * np.sin(freq * x)).mean(axis=2),
        )

        plt.clf()
        plt.imshow(magnitude)
        plt.title(f"Frequency: {freq:.2f}")
        plt.pause(0.5)
        #plt.show()

        freq *= 1.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()

    dataset = WDDDataset(args.data)
    print(f"Number of samples: {len(dataset)}")

    i = random.randint(0, len(dataset) - 1)
    print(f"Sampled index: {i}")
    imgs = dataset[i]
    print(f"Shape: {imgs.shape}")

    #print("Showing sample.")
    #show_sample(imgs)

    print("Plotting changing DFT.")
    plot_changing_dft(imgs)


if __name__ == "__main__":
    main()
