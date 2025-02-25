"""
Script to allow manual labeling.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pygame

VALID_EXTS = (".png", ".jpg", ".jpeg")
RES = 800
BRUSH_SIZE = 10


def label_img(window, args, img_path, i):
    """
    Returns False if the user wants to quit.
    """
    img = pygame.image.load(str(img_path))
    orig_res = img.get_size()
    img = pygame.transform.scale(img, (RES, RES))
    label_img = np.zeros((RES, RES), dtype=np.uint8)

    window.blit(img, (0, 0))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    img = pygame.surfarray.array3d(window)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, orig_res)
                    label_img = cv2.resize(label_img, orig_res)
                    shutil.copy(img_path, args.output / f"{i}_img.jpg")
                    cv2.imwrite(str(args.output / f"{i}_label.jpg"), label_img)
                    return True
                elif event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_r:
                    label_img = np.zeros((RES, RES), dtype=np.uint8)
                    window.blit(img, (0, 0))

        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(window, (0, 255, 0), (x, y), BRUSH_SIZE)
            cv2.circle(label_img, (x, y), BRUSH_SIZE, 255, -1)

        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)

    window = pygame.display.set_mode((RES, RES))

    i = 0
    for file in args.data.glob("**/*"):
        if not file.is_file():
            continue
        if file.suffix not in VALID_EXTS:
            print("Invalid file extension, skipping", file)
            continue

        print("Labeling", file)

        if not label_img(window, args, file, i):
            break
        i += 1


if __name__ == "__main__":
    main()
