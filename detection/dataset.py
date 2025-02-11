"""
PyTorch dataset class. Handles parsing dataset from disk.
"""

import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image


# Crop resolution in raw image.
BLOCK_SIZE = 512
# Block indices (x, y).
VALID_BLOCKS = []
for x in range(2, 8):
    for y in range(4, 8):
        VALID_BLOCKS.append((x, y))
# Output res of dataset.
OUTPUT_RES = 256

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
)


class OistDataset(Dataset):
    """
    Iterates dataset from https://groups.oist.jp/bptu/honeybee-tracking-dataset

    Each file is (1024 * 5 = 5120px) pixels square.
    Split this into 25 blocks of 1024px.
    In some datasets, only certain blocks (VALID_BLOCKS) are annotated.

    The ground truth is an ellipse mask at the position and orientation of each bee,
    according to the 2017 paper.

    Will call self.cache_ground_truth() to write new cache files to disk.
    """

    def __init__(self, dir, load_memory=False):
        """
        load_memory: Whether to load all images into memory (beware memory usage).
        """
        self.dir = Path(dir)
        self.load_memory = load_memory
        self.memory_cache = None

        assert (self.dir / "frames").is_dir()
        assert (self.dir / "frames_txt").is_dir()
        (self.dir / "ground_truth").mkdir(exist_ok=True)

        self.files = list((self.dir / "frames").iterdir())
        self.files = [f for f in self.files if f.suffix in VALID_EXTENSIONS]

        self.resize = T.Resize((OUTPUT_RES, OUTPUT_RES), antialias=True)
        # self.blur = T.GaussianBlur(5)

        self.cache_ground_truth()
        if self.load_memory:
            self.do_load_memory()

    def get_gt_path(self, file_idx):
        return (self.dir / "ground_truth" / self.files[file_idx].stem).with_suffix(".jpg")

    def __len__(self):
        return len(self.files) * len(VALID_BLOCKS)

    def __getitem__(self, idx):
        """
        return (x, y)

        x is image.
            Tensor shape (1, BLOCK_SIZE, BLOCK_SIZE), dtype uint8

        y is an image same shape and dtype as x.
        Ellipse masks are drawn at a color value of 255,
            corresponding to the body of the bees, as described in the paper.
        """
        file_idx = idx // len(VALID_BLOCKS)
        block_x, block_y = VALID_BLOCKS[idx % len(VALID_BLOCKS)]
        # min/max pixels of the block.
        min_x = block_x * BLOCK_SIZE
        min_y = block_y * BLOCK_SIZE
        max_x = min_x + BLOCK_SIZE
        max_y = min_y + BLOCK_SIZE

        if self.load_memory:
            x, y = self.memory_cache[idx]
        else:
            x = read_image(str(self.files[file_idx]))[..., min_y:max_y, min_x:max_x]
            y = read_image(str(self.get_gt_path(file_idx)))[..., min_y:max_y, min_x:max_x]

        x = self.resize(x)
        y = self.resize(y)
        return x, y

    @staticmethod
    def read_label(path):
        with open(path, "r") as f:
            for line in f:
                values = line.split()
                values = [*map(int, values[:-1]), float(values[-1])]
                cls = values[2]
                x = values[0] + values[3]
                y = values[1] + values[4]
                angle = values[5]
                yield (cls, x, y, angle)

    def draw_ground_truth(self, label_path, height, width):
        """
        Draw ground truth (ellipses mask) for one sample, and return as np array.
        """
        # Draw ellipses.
        y_img = np.zeros((height, width, 1), dtype=np.uint8)
        for _, x, y, angle in OistDataset.read_label(label_path):
            angle = angle * 180 / math.pi + 90
            cv2.ellipse(y_img, (x, y), (35, 20), angle, 0, 360, 255, -1)

        return y_img

    def cache_ground_truth(self, force_rebuild=False):
        """
        Call this before training to generate ground truth cache.
        """
        for i in trange(len(self.files), desc="Generating ground truth cache"):
            gt_path = self.get_gt_path(i)
            if not gt_path.exists() or force_rebuild:
                x = read_image(str(self.files[i]))
                label_path = (self.dir / "frames_txt" / self.files[i].stem).with_suffix(".txt")
                y = self.draw_ground_truth(label_path, x.shape[1], x.shape[2])
                cv2.imwrite(str(gt_path), y)

    def do_load_memory(self):
        """
        Load all images into memory.
        """
        self.memory_cache = []
        for i in trange(len(self.files), desc="Loading dataset into memory"):
            gt_path = self.get_gt_path(i)
            x = read_image(str(self.files[i]))
            y = read_image(str(gt_path))
            self.memory_cache.append((x, y))


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    dataset = OistDataset(args.data)

    time_start = time.time()
    for i in range(10):
        _ = dataset[i]
    time_end = time.time()
    print("Fetched 10 samples in", time_end - time_start)

    x, y = dataset[10]
    cv2.imwrite("x.png", x.permute(1, 2, 0).numpy())
    cv2.imwrite("y.png", y.permute(1, 2, 0).numpy())
