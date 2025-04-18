"""
Global constants.
"""

import torch

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
    from dataset import OistDataset, ThoraxDataset, VarroaDataset
    from model_dino import DinoNN
    from model_unet import ReducedUNet

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


def extract_blobs(pred, threshold=0.5, blur=0) -> list[torch.Tensor]:
    """
    Extract blobs from a prediction map.
    Args:
        pred: (H, W) [0, 1] prediction
        threshold: threshold for pixel being counted.
        blur: If nonzero, run Gaussian blur with this kernel size before processing.
    Returns:
        list of (H, W) bool tensors, each one a unique contiguous region.
            Obtaining size of blob and mean coordinates is trivial.
    """
    pred = pred.clone()
    if blur > 0:
        pred = torch.nn.functional.gaussian_blur(pred, kernel_size=blur)
    pred = pred > threshold

    blobs = []
    while pred.any():
        y, x = torch.where(pred)
        y, x = y[0], x[0]

        blob = torch.zeros_like(pred, dtype=torch.bool)
        stack = [(y, x)]
        while stack:
            y, x = stack.pop()
            blob[y, x] = True
            pred[y, x] = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < pred.shape[0] and 0 <= nx < pred.shape[1]:
                        if pred[ny, nx]:
                            stack.append((ny, nx))

        blobs.append(blob)

    return blobs
