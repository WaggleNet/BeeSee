from pathlib import Path

import cv2
import numpy as np


class WDDDataset:
    def __init__(self, path: Path):
        self.path = path
        self.samples = []
        for folder in path.glob("**/"):
            if folder.is_dir() and len(folder.stem) == 1:
                self.samples.append(folder)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        files = []
        for f in self.samples[index].iterdir():
            if f.suffix == ".png" and f.stem.startswith("image_"):
                num = int(f.stem.split("_")[-1])
                files.append((num, f))
        files.sort(key=lambda x: x[0])
        files = [x[1] for x in files]

        imgs = [cv2.imread(str(f))[..., 0] for f in files]
        imgs = np.stack(imgs, axis=0)
        imgs = imgs.astype(np.float32) / 255.0

        return imgs
