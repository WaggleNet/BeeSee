"""
Global constants.
"""

import torch

from dataset import *
from model_dino import DinoNN
from model_unet import ReducedUNet

MODEL_CHOICES = (
    "unet",
    "dino",
)
DATA_CHOICES = (
    "oist",
    "thorax",
    "varroa",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset_model(data_type, model_type):
    """
    Returns dataset and model class based on args.data_type and args.model_type.
    If either is None, the respective return value is None.

    Return: dataset_cls, model_cls, dataset_kwargs
    """
    dataset_args = {}
    if model_type == "dino":
        dataset_args["y_patches"] = True

    if data_type is None:
        dataset = None
    else:
        if data_type == "oist":
            dataset = OistDataset
        elif data_type == "thorax":
            dataset = ThoraxDataset
        elif data_type == "varroa":
            dataset = VarroaDataset
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    if model_type is None:
        model = None
    else:
        if model_type == "unet":
            model = ReducedUNet
        elif model_type == "dino":
            model = DinoNN
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return dataset, model, dataset_args
