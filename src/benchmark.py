"""
Benchmark model performance.
E.g. test quantization performance.
"""

import argparse
import time

import cv2
import torch
from utils import DEVICE, load_dino_model, preprocess_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--img", required=True)
    args = parser.parse_args()

    model = load_dino_model(args.model)
    model.eval()

    img = cv2.imread(args.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    img = preprocess_images(img, res=224)[0]
    img = img.unsqueeze(0).to(DEVICE)

    # Warm up
    model(img)
    # Test at least 5 seconds, at least 2 iters.
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


if __name__ == "__main__":
    main()
