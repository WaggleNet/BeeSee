"""
Test PCA on output of 2D DFT.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

from dft_2d import compute_spatial_dft
from perpixel_dft import get_sample, compute_dft


def compute_pca(pts):
    u, s, v = scipy.linalg.svd(pts, full_matrices=False)
    return v[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    imgs = get_sample(args.data)
    dft = compute_dft(imgs)
    spatial_dft = compute_spatial_dft(dft)

    # Get pixels above threshold.
    img = np.interp(spatial_dft, (spatial_dft.min(), spatial_dft.max()), (0, 1))
    img = img > 0.5
    pts = np.argwhere(img)
    pts = (pts - img.shape[0] / 2) / (img.shape[0] / 2)

    pc = compute_pca(pts)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(spatial_dft)
    ax[1].imshow(img)
    ax[1].axline((img.shape[1] // 2, img.shape[0] // 2), slope=pc[0] / pc[1], color="red", label="PCA")

    plt.show()


if __name__ == "__main__":
    main()
