"""
Run thorax detection on frames of a video and save result.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from tqdm import tqdm

from model_dino import DinoNN
from utils import DEVICE, extract_blobs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    model = DinoNN().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    video_read = cv2.VideoCapture(str(args.video))
    video_write = cv2.VideoWriter(
        "test_thorax.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_read.get(cv2.CAP_PROP_FPS),
        (int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )

    pbar = tqdm(total=int(video_read.get(cv2.CAP_PROP_FRAME_COUNT) / args.step))
    i = 0
    while True:
        for _ in range(args.step):
            ret, frame = video_read.read()
        if not ret:
            break

        # Crop to square
        if frame.shape[0] > frame.shape[1]:
            frame = frame.swapaxes(0, 1)
        x_extra = (frame.shape[1] - frame.shape[0]) // 2
        frame = frame[:, x_extra:-x_extra, :]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).to(DEVICE)
        frame = frame.float() / 255.0
        frame = frame.permute(2, 0, 1)
        frame = frame.mean(dim=0, keepdim=True)
        frame = frame.unsqueeze(0)
        frame = F.resize(frame, (448, 448), antialias=True)

        pred = model(frame).squeeze(0)
        pred = F.resize(pred, (frame.shape[2], frame.shape[3]), antialias=True)
        pred = pred.squeeze(0)

        frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frame[pred > 0.5] = 255

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_write.write(frame)

        cv2.imwrite(f"{i}.jpg", frame)
        i += 1

        pbar.update(1)

    pbar.close()
    video_read.release()
    video_write.release()


if __name__ == "__main__":
    main()
