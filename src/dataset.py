"""
Implementation of different datasets.
Datasets specific to HoneyBee, OIST, or other extend from the base class.
"""

import math
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image

from utils import *


class BeeSeeDataset(Dataset):
    """
    Base dataset class.

    Each data is a input-label pair:
        Input: Image, dtype float, 0-1, shape (1, H, W)
            I.e. image of bee(s).
        Label: Image, dtype float, 0-1, shape (1, H, W)
            Can be different resolution to input; e.g. DINO outputs input_res // 14.
            Label semantic meaning is dataset-specific, e.g. bee or thorax position.
    """

    def __init__(self, dir, output_res=448, y_patches=False, **kwargs):
        """
        dir: Path to dataset directory.
        output_res: Resolution of output x and y image.
        y_patches: If True, y image is scaled down by 14x (used for DINO and other ViT).
        """
        self.dir = dir
        self.output_res = output_res

        self.trans_both = T.Compose([
            T.Resize((self.output_res, self.output_res)),
            T.RandomResizedCrop(self.output_res, scale=(0.6, 1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=360),
        ])
        self.trans_x = T.Compose([
            T.RandomAdjustSharpness(0.7),
        ])
        trans_y = [T.ElasticTransform(25.0)]
        if y_patches:
            trans_y.append(T.Resize((self.output_res // 14, self.output_res // 14)))
        self.trans_y = T.Compose(trans_y)

    def apply_transforms(self, x, y):
        x_ch = x.shape[0]
        both = torch.cat((x, y), dim=0)
        both = self.trans_both(both)
        x, y = both[:x_ch], both[x_ch:]
        x = self.trans_x(x)
        y = self.trans_y(y)
        return x, y

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class OistDataset(BeeSeeDataset):
    """
    Iterates OIST bee detection dataset.

    Each image is (5 * 1024 = 5120px) square.
    We split this into 25 blocks of 1024px.
    Only a subset of those blocks (VALID_BLOCKS) are annotated and used.

    The ground truth is an ellipse mask at the position and orientation of each bee,
    according to the 2017 paper.
    """

    BLOCK_SIZE = 1024
    VALID_BLOCKS = []
    for x in range(1, 4):
        for y in range(2, 4):
            VALID_BLOCKS.append((x, y))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert (self.dir / "frames").is_dir()
        assert (self.dir / "frames_txt").is_dir()

        self.files = list((self.dir / "frames").iterdir())
        self.files = [f for f in self.files if f.suffix == ".jpg"]

    def __len__(self):
        return len(self.files) * len(self.VALID_BLOCKS)

    def __getitem__(self, idx):
        """
        x: Image of beehive, cropped to corresponding block.
        y: Image of location of bee bodies.
            Ellipses are drawn on top of every bee.
        """
        file_idx = idx // len(self.VALID_BLOCKS)
        block_x, block_y = self.VALID_BLOCKS[idx % len(self.VALID_BLOCKS)]
        # min/max pixels of the block.
        min_x = block_x * self.BLOCK_SIZE
        min_y = block_y * self.BLOCK_SIZE
        max_x = min_x + self.BLOCK_SIZE
        max_y = min_y + self.BLOCK_SIZE

        x_img = read_image(str(self.files[file_idx]))[..., min_y:max_y, min_x:max_x]
        x_img = x_img.float() / 255

        label_path = (self.dir / "frames_txt" / self.files[file_idx].name).with_suffix(".txt")
        labels = []
        # Filter labels in block.
        for label in OistDataset.read_label(label_path):
            if (min_x < label[1] < max_x
                    and min_y < label[2] < max_y
                    and label[0] == 1):
                labels.append((label[1] - min_x, label[2] - min_y, label[3]))

        y_img = OistDataset.draw_label(x_img.shape, labels)

        x_img, y_img = self.apply_transforms(x_img, y_img)
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

    @staticmethod
    def draw_label(shape, labels):
        """
        shape: Label image shape (i.e. x_img.shape).
        labels: List of (x, y, angle) tuples, relative to this patch.
        return: float, 0-1
        """
        # Draw ellipse.
        y_img = np.zeros((shape[1], shape[2], 1), dtype=np.uint8)
        for x, y, angle in labels:
            angle = angle * 180 / math.pi + 90
            cv2.ellipse(y_img, (x, y), (35, 20), angle, 0, 360, 255, -1)
        y_img = torch.from_numpy(y_img).permute(2, 0, 1)
        y_img = y_img.float() / 255
        return y_img


class ThoraxDataset(BeeSeeDataset):
    """
    Iterates the HoneyBee dataset with thorax annotations.
    """

    def __init__(self, duplicity=10, **kwargs):
        super().__init__(**kwargs)
        self.duplicity = duplicity

        self.indices = set()
        for f in self.dir.iterdir():
            if "_img" in f.stem:
                self.indices.add(f.stem.split("_")[0])
        self.indices = list(self.indices)

    def __len__(self):
        return len(self.indices) * self.duplicity
    
    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        idx = idx % len(self.indices)
        x = read_image(str(self.dir / f"{self.indices[idx]}_img.jpg")).float() / 255
        x = torch.mean(x, dim=0, keepdim=True)  # Convert to grayscale.
        y = read_image(str(self.dir / f"{self.indices[idx]}_label.jpg")).float() / 255
        x, y = self.apply_transforms(x, y)
        return x, y


class VarroaDataset(BeeSeeDataset):
    """
    Varroa destructor dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert (self.dir / "images").is_dir()
        assert (self.dir / "labels").is_dir()

        self.files = list((self.dir / "images").iterdir())
        self.files = [f for f in self.files if f.suffix in (".jpg", ".png")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        x: Image of bee.
        y: Image of location of varroa, if exists.
        """
        x_img = read_image(str(self.files[idx])).float() / 255
        x_img = torch.mean(x_img, dim=0, keepdim=True)

        label_path = (self.dir / "labels" / self.files[idx].name).with_suffix(".txt")
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                label = list(map(float, line.split()))[1:]
                labels.append(label)
        labels = np.array(labels)

        y_img = VarroaDataset.draw_label(x_img.shape, labels)

        x_img, y_img = self.apply_transforms(x_img, y_img)
        return x_img, y_img

    @staticmethod
    def draw_label(shape, labels):
        """
        shape: Label image shape (i.e. x_img.shape).
        labels: Tensor shape (N, 4). Each label is (x, y, w, h) in YOLOv5 format.
            x, y is center. w, h, is width, height.
            0-1, factor of image size.
        return: float, 0-1
        """
        y_img = np.zeros((shape[1], shape[2], 1), dtype=np.uint8)

        if len(labels) > 0:
            labels[:, 0] -= labels[:, 2] / 2
            labels[:, 1] -= labels[:, 3] / 2
            labels[:, 0] *= shape[2]
            labels[:, 1] *= shape[1]
            labels[:, 2] *= shape[2]
            labels[:, 3] *= shape[1]
            labels = labels.astype(int)

            for x, y, w, h in labels:
                cv2.rectangle(y_img, (x, y), (x + w, y + h), 255, -1)

        y_img = torch.from_numpy(y_img).permute(2, 0, 1)
        y_img = y_img.float() / 255
        return y_img


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data_type", choices=DATA_CHOICES, required=True)
    args = parser.parse_args()

    dataset_cls, _, dataset_args = get_dataset_model(args.data_type, None)
    dataset = dataset_cls(dir=args.data, **dataset_args)

    # Draw 3 by 6 plot of X Y samples.
    indices = random.sample(range(len(dataset)), 6)
    fig, axs = plt.subplots(3, 6, dpi=300)
    axs[0, 0].set_title("X")
    axs[1, 0].set_title("Y")
    axs[2, 0].set_title("X + Y")
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        axs[0, i].imshow(x[0], cmap="gray")
        axs[1, i].imshow(y[0], cmap="gray")
        x[y > 0.5] = 1
        axs[2, i].imshow(x[0], cmap="gray")
        for j in range(3):
            axs[j, i].axis("off")

    x, y = dataset[0]
    print("X", x.shape, x.min(), x.max())
    print("Y", y.shape, y.min(), y.max())

    #plt.show()
    plt.savefig("dataset.jpg")
