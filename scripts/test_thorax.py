"""
Run thorax detection on frames of a video and save result.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse

import cv2
import torch
import torchvision.transforms.functional as F

from model_dino import DinoNN
from utils import DEVICE, extract_blobs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Omit to use camera.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model = DinoNN().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    vid_path = 0 if args.video is None else args.video
    video_read = cv2.VideoCapture(vid_path)
    if vid_path == 0:
        video_read.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
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
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame[pred > 0.5] = (0, 255, 0)

        """
        blobs = extract_blobs(pred, 0.5)
        print(f"Found {len(blobs)} blobs: ", end="")
        for b in blobs:
            # Extract COM
            y, x = torch.where(b)
            com = (x.float().mean(), y.float().mean())
            print(f"{com} ", end="")
        print()
        """

        cv2.imshow("a", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    video_read.release()


if __name__ == "__main__":
    main()
