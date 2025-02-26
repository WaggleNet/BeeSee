from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image


class ThoraxDataset(Dataset):
    def __init__(self, data_dir: Path, res=448, duplicity=10):
        assert res % 14 == 0

        self.data_dir = data_dir
        self.indices = set()
        for f in data_dir.iterdir():
            if "_img" in f.stem:
                self.indices.add(f.stem.split("_")[0])
        self.indices = list(self.indices)

        self.res = res
        self.duplicity = duplicity

        self.transform_both = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(360),
            T.RandomResizedCrop(res, scale=(0.5, 1.0)),
        ])
        self.transform_x = T.Compose([
            T.Resize((res, res)),
            T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.6, hue=0.1),
            T.ElasticTransform(10.0),
            T.RandomInvert(0.3),
        ])
        self.transform_y = T.Compose([
            T.Resize((res // 14, res // 14)),
            T.ElasticTransform(),
        ])

    def __len__(self):
        return len(self.indices) * self.duplicity

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        idx = idx % len(self.indices)

        img = read_image(str(self.data_dir / f"{self.indices[idx]}_img.jpg")).float() / 255
        label = read_image(str(self.data_dir / f"{self.indices[idx]}_label.jpg")).float() / 255

        both = torch.cat([img, label], dim=0)
        both = self.transform_both(both)
        img, label = both[:3], both[3:]
        img = self.transform_x(img)
        label = self.transform_y(label)
        mask = label > 0.5
        label[mask] = 1
        label[~mask] = 0

        return img, label
