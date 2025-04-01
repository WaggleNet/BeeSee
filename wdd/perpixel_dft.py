"""
Test per pixel over time DFT.
"""

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


def compute_dft(imgs, freq):
    x = np.arange(imgs.shape[2])
    magnitude = np.hypot(
        (imgs * np.cos(freq * x)).mean(axis=2),
        (imgs * np.sin(freq * x)).mean(axis=2),
    )
    return magnitude


def plot_changing_dft(imgs):
    """
    Each time series is single pixel over time, i.e. imgs[:, i, j]

    imgs: (T, H, W)
    """
    freq = 2 * math.pi / imgs.shape[2]
    while freq <= math.pi:
        dft = compute_dft(imgs, freq)
        plt.clf()
        plt.imshow(dft)
        plt.title(f"Frequency: {freq:.2f}")
        plt.pause(0.5)
        freq *= 1.1


def get_sample(path):
    """
    Returns imgs as (H, W, T) (i.e. does a swapaxes).
    """
    dataset = WDDDataset(path)
    print(f"Number of samples: {len(dataset)}")

    i = random.randint(0, len(dataset) - 1)
    print(f"Sampled index: {i}")
    imgs = dataset[i]
    print(f"Shape: {imgs.shape}")

    imgs = imgs.transpose(1, 2, 0)  # (H, W, T)
    imgs = imgs * 2 - 1

    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freq", type=float)
    args = parser.parse_args()

    imgs = get_sample(args.data)

    #print("Showing sample.")
    #show_sample(imgs)

    if args.freq is None:
        print("Plotting changing DFT.")
        plot_changing_dft(imgs)
    else:
        print("Plotting DFT, freq =", args.freq)
        dft = compute_dft(imgs, args.freq)
        plt.imshow(dft)
        plt.show()


if __name__ == "__main__":
    main()
