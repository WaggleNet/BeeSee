import argparse
import random

import cv2
import numpy as np

import torch
from torchvision import transforms as T
from torchvision.io import read_image

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, required=True, help="Path to the image file.")
parser.add_argument("--res", type=int, default=896)
args = parser.parse_args()

resize_f = T.Resize((args.res, args.res))
img = read_image(args.img)
img = (img.float() / 255).unsqueeze(0)
img = resize_f(img)
print("img shape:", img.shape)

random.seed(1)
torch.random.manual_seed(1)

layers = dino.get_intermediate_layers(img, n=8, reshape=True)
print("Number of hidden layers:", len(layers))
for layer in layers:
    print("Layer shape:", layer.shape)


def show_attn(attn):
    attn = attn.detach().numpy()[0]

    orig_res = attn.shape[1:]
    indices = random.choices(range(orig_res[0] * orig_res[1]), k=512)
    mat = attn.reshape(attn.shape[0], -1)[:, indices].T
    u, s, v = np.linalg.svd(mat)
    comps = v[:, :3]

    img = np.zeros((3, orig_res[0], orig_res[1]))
    for i in range(3):
        img[i] = np.tensordot(attn, comps[:, i], axes=(0, 0))
    img = img.transpose(1, 2, 0).astype(np.float32)
    img = 1 / (1 + np.exp(-img))

    img = (img * 255).astype(np.uint8)
    return img


img = cv2.imread(args.img)
for i, layer in enumerate(layers):
    attn = show_attn(layer)
    cv2.imwrite(f"attn{i}.jpg", attn)
