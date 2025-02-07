import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset directory.")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
