"""
DINO, then conv head taking in hidden layers.
"""

import argparse

import torch
import torch.nn as nn

from utils import DEVICE

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(DEVICE)
DINO.eval()


class DinoNN(nn.Module):
    def __init__(self, num_hidden_layers=2):
        """
        num_hidden_layers: Number of hidden layers to use from DINO.
        """
        super().__init__()

        self.num_hidden_layers = num_hidden_layers

        self.head = nn.Sequential(
            nn.Conv2d(384 * self.num_hidden_layers, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        """
        x: (B, 1, H, W), float 0-1.
        return: (B, 1, H, W), float 0-1.
        """
        with torch.no_grad():
            x = torch.cat([x, x, x], dim=1)
            x = DINO.get_intermediate_layers(x, n=self.num_hidden_layers, reshape=True)
            x = torch.cat(x, dim=1)

        x = self.head(x)
        if not logits:
            x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    # Benchmark model performance.
    import cv2
    import time
    from utils import load_dino_model, preprocess_images

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--img", required=True)
    args = parser.parse_args()

    model = load_dino_model(args.model)
    model.eval()
    img = cv2.imread(args.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    img = preprocess_images(img, res=448)[0]
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        time_start = time.time()
        iters = 0
        while True:
            pred = model(img)

            iters += 1
            if iters >= 2 and time.time() - time_start > 5:
                break

    elapse = time.time() - time_start
    print(f"DINO performance test:")
    print(f"  iters: {iters}")
    print(f"  elapse: {elapse:.2f}s")
    print(f"  elapse_per: {elapse / iters:.3f}s")
    print(f"  fps: {iters / elapse:.3f}")

    # Make and save prediction.
    img = cv2.imread(args.img)
    pred = pred.detach().cpu().numpy()[0, 0]
    pred = cv2.resize(pred, (img.shape[1], img.shape[0]))
    img[pred > 0.5] = (0, 255, 0)
    cv2.imwrite("dino_perft_pred.jpg", img)
