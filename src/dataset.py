from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image

RES = 128


class ImageDataset(Dataset):
    """
    Iterates all jpeg images in a directory.
    """

    def __init__(self, dir: Path):
        self.dir = dir
        self.files = list(dir.glob("*.jpg"))

        self.transform = T.Compose([
            T.Resize((RES, RES)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Reads image as tensor.
        Shape: (3, RES, RES)
        Dtype: float, range [0, 1]
        """
        img = read_image(self.files[idx])
        img = img.float() / 255.0
        img = self.transform(img)
        return img
