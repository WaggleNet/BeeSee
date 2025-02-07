import argparse

import cv2
import numpy as np

import torch
from torchvision import transforms as T
from torchvision.io import read_image

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, required=True, help="Path to the image file.")
parser.add_argument("--res", type=int, default=448)
args = parser.parse_args()

resize_f = T.Resize((args.res, args.res))
img = read_image(args.img)
img = (img.float() / 255).unsqueeze(0)
img = resize_f(img)
print("img shape:", img.shape)

layers = dino.get_intermediate_layers(img, n=2, reshape=True)
print("Number of hidden layers:", len(layers))
for layer in layers:
    print("Layer shape:", layer.shape)


def show_attn(attn, win_name):
    img = attn.detach().numpy()[0]
    img = np.linalg.norm(img, axis=0)
    img = img / img.max() * 255
    img = img.astype("uint8")
    cv2.imshow(win_name, img)


img = cv2.imread(args.img)
cv2.imshow("img", img)
for i, layer in enumerate(layers):
    show_attn(layer, f"attn {i}")

while True:
    if cv2.waitKey(1) == ord("q"):
        break
