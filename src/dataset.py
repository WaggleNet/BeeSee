import math
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image


BLOCK_SIZE = 1024
# Block indices (x, y) for the 30FPS dataset.
VALID_BLOCKS = (
    (1, 2),
    (2, 2),
    (3, 2),
    (1, 3),
    (2, 3),
    (3, 3),
)
OUTPUT_RES = 512

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
)


class OistDataset(Dataset):
    """
    Iterates dataset from https://groups.oist.jp/bptu/honeybee-tracking-dataset

    Each file is (1024 * 5 = 5120) pixels square.
    Split this into 25 grids of 1024.
    In some datasets, only certain grids are annotated.
    """

    def __init__(self, dir):
        self.dir = Path(dir)
        assert (self.dir / "frames").is_dir()
        assert (self.dir / "frames_txt").is_dir()

        self.files = list((self.dir / "frames").iterdir())
        self.files = [f for f in self.files if f.suffix in VALID_EXTENSIONS]

        self.resize = T.Resize((OUTPUT_RES, OUTPUT_RES), antialias=True)
        self.blur = T.GaussianBlur(5)

    def __len__(self):
        return len(self.files) * len(VALID_BLOCKS)

    def __getitem__(self, idx):
        """
        return (x, y)

        x is image.
            Tensor shape (1, BLOCK_SIZE, BLOCK_SIZE), dtype uint8

        y is an image same shape and dtype as x.
        Ellipses are drawn at a color value of 255,
            corresponding to the body of the bees, as described in the paper.
        """
        file_idx = idx // len(VALID_BLOCKS)
        block_x, block_y = VALID_BLOCKS[file_idx % len(VALID_BLOCKS)]
        min_x = block_x * BLOCK_SIZE
        min_y = block_y * BLOCK_SIZE
        max_x = min_x + BLOCK_SIZE
        max_y = min_y + BLOCK_SIZE

        img = read_image(str(self.files[file_idx]))[..., min_y:max_y, min_x:max_x]

        label_path = (self.dir / "frames_txt" / self.files[file_idx].stem).with_suffix(".txt")
        labels = []
        for label in OistDataset.read_label(label_path):
            if (min_x < label[1] < max_x
                    and min_y < label[2] < max_y
                    and label[0] == 1):
                labels.append((label[1] - min_x, label[2] - min_y, label[3]))

        y_img = np.zeros((img.shape[1], img.shape[2], 1), dtype=np.uint8)
        for x, y, angle in labels:
            angle = angle * 180 / math.pi + 90
            cv2.ellipse(y_img, (x, y), (35, 20), angle, 0, 360, 255, -1)
        y_img = torch.from_numpy(y_img).permute(2, 0, 1)

        img = self.resize(img)
        y_img = self.blur(self.resize(y_img))
        return img, y_img

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
    dataset = OistDataset("/mnt/data/patrick/Datasets/OistBee/Detection")
    x, y = dataset[10]
    cv2.imwrite("x.png", x.permute(1, 2, 0).numpy())
    cv2.imwrite("y.png", y.permute(1, 2, 0).numpy())
