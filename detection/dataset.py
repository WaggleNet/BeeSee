"""
PyTorch dataset class. Handles parsing dataset from disk.
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np

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
    """

    def __init__(self, dir, do_aug=True):
        """
        load_memory: Whether to load all images into memory (beware memory usage).
        """
        self.dir = Path(dir)
        self.do_aug = do_aug

        assert (self.dir / "frames").is_dir()
        assert (self.dir / "frames_txt").is_dir()
        (self.dir / "ground_truth").mkdir(exist_ok=True)

        self.files = list((self.dir / "frames").iterdir())
        self.files = [f for f in self.files if f.suffix in VALID_EXTENSIONS]

        self.transform = T.Compose([
            T.Resize((OUTPUT_RES, OUTPUT_RES)),
        ])
        self.both_aug = T.Compose([
            T.RandomResizedCrop(OUTPUT_RES, scale=(0.6, 1)),
        ])
        self.x_aug = T.Compose([
            T.RandomAdjustSharpness(0.7),
        ])
        self.y_aug = T.Compose([
            T.ElasticTransform(),
        ])

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

        x_img = read_image(str(self.files[file_idx]))[..., min_y:max_y, min_x:max_x]

        label_path = (self.dir / "frames_txt" / self.files[file_idx].stem).with_suffix(".txt")
        labels = []
        # Filter labels in block.
        for label in OistDataset.read_label(label_path):
            if (min_x < label[1] < max_x
                    and min_y < label[2] < max_y
                    and label[0] == 1):
                labels.append((label[1] - min_x, label[2] - min_y, label[3]))

        # Draw ellipse.
        y_img = np.zeros((x_img.shape[1], x_img.shape[2], 1), dtype=np.uint8)
        for x, y, angle in labels:
            # if random.random() < 0.5:
            #     continue
            angle = angle * 180 / math.pi + 90
            cv2.ellipse(y_img, (x, y), (35, 20), angle, 0, 360, 255, -1)
        y_img = torch.from_numpy(y_img).permute(2, 0, 1)

        both_img = self.transform(torch.stack([x_img, y_img], dim=0))
        if self.do_aug and random.random() < 0.5:
            both_img = self.both_aug(both_img)
            x_img, y_img = both_img[0], both_img[1]
            x_img = self.x_aug(x_img)
            y_img = self.y_aug(y_img)
        else:
            x_img, y_img = both_img[0], both_img[1]
        return x_img, y_img

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
