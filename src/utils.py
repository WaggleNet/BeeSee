"""
Global constants.
"""

import torch

from dataset import *
from model_dino import DinoNN
from model_unet import ReducedUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset_model(data_type, model_type):
    """
    Returns dataset and model class based on args.data_type and args.model_type.
    If either is None, the respective return value is None.
    """
    if data_type is None:
        dataset = None
    else:
        if data_type == "oist":
            dataset = OistDataset
        elif data_type == "thorax":
            dataset = ThoraxDataset
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    if model_type is None:
        model = None
    else:
        if model_type == "unet":
            model = ReducedUNet
        elif model_type == "dino":
            return DinoNN
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return dataset, model
