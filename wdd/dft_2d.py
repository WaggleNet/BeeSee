"""
Test 2D DFT on output of per-pixel DFT, to find direction and length of waggle.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perpixel_dft import get_sample, compute_dft


def compute_spatial_dft(dft, freq_max, bins, omit_thres=math.pi / 15):
    res = np.zeros((bins, bins), dtype=float)
    x, y = np.meshgrid(np.arange(dft.shape[1]), np.arange(dft.shape[0]))

    for i in range(bins):
        for j in range(bins):
            w_x = np.interp(j, (0, bins), (-freq_max, freq_max))
            w_y = np.interp(i, (0, bins), (-freq_max, freq_max))
            if abs(w_x) < omit_thres and abs(w_y) < omit_thres:
                # Omit low freqs.
                continue

            theta = x * w_x + y * w_y
            out = np.hypot(
                (dft * np.cos(theta)).mean(),
                (dft * np.sin(theta)).mean(),
            )
            res[i, j] = out

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freq_time", type=float, default=0.75)
    parser.add_argument("--freq_space_max", type=float, default=math.pi)
    parser.add_argument("--freq_space_bins", type=int, default=50)
    args = parser.parse_args()

    imgs = get_sample(args.data)
    dft = compute_dft(imgs, args.freq_time)
    spatial_dft = compute_spatial_dft(dft, args.freq_space_max, args.freq_space_bins)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dft)
    ax[0].set_title("Temporal DFT")
    ax[1].imshow(spatial_dft)
    ax[1].set_title("Spatial DFT on Temporal DFT")
    plt.show()


if __name__ == "__main__":
    main()
