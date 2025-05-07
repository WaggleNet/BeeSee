"""
Run thorax detection on frames of a video and save result.
"""

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from utils import extract_blobs, load_dino_model, preprocess_images

COMPUTE_BLOBS = False
USE_PICAM = False

if USE_PICAM:
    from picamera2 import Picamera2


def run_thorax_model(model, frame):
    """
    Run thorax detection on cv2 image (np array, HWC, uint8).

    return (frame, pred)
        frame is the cropped cv2 image.
        pred is detected mask (H, W)
    """
    # Crop to square
    if frame.shape[0] > frame.shape[1]:
        frame = frame.swapaxes(0, 1)
    x_extra = (frame.shape[1] - frame.shape[0]) // 2
    frame = frame[:, x_extra:-x_extra, :]

    frame = preprocess_images(frame)

    pred = model(frame).squeeze(0)
    pred = F.resize(pred, (frame.shape[2], frame.shape[3]), antialias=True)
    pred = pred.squeeze(0)
    pred = pred > 0.5
    pred = pred.cpu().numpy()
    # pred: (H, W), int, np array

    # Turn frame back into np array
    frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame, pred


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Omit to use camera.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model = load_dino_model(args.model)

    # Setup camera or video stream.
    if USE_PICAM:
        print("Using PiCam.")
        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main={
            "format": "RGB888",
            "size": (1280, 720),
        }))
        picam.start()
        time.sleep(1)

    else:
        vid_path = 0 if args.video is None else args.video
        video_read = cv2.VideoCapture(vid_path)
        if vid_path == 0:
            video_read.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Using opencv video reader, path =", vid_path)

    while True:
        if USE_PICAM:
            frame = picam.capture_array()
        else:
            ret, frame = video_read.read()
            if not ret:
                break

        frame, pred = run_thorax_model(model, frame)
        frame[pred] = (0, 255, 0)

        if COMPUTE_BLOBS:
            blobs = extract_blobs(pred, 0.5)
            print(f"Found {len(blobs)} blobs: ", end="")
            for b in blobs:
                # Extract COM
                y, x = torch.where(b)
                com = (x.float().mean(), y.float().mean())
                print(f"{com} ", end="")
            print()

        cv2.imshow("a", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    if not USE_PICAM:
        video_read.release()


if __name__ == "__main__":
    main()
