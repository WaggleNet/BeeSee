"""
Run before: ./test_gphoto.sh

This script just streams from a video file.
"""
# gphoto2 --stdout --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -f v4l2 /dev/video11

import cv2


video = cv2.VideoCapture("/dev/video11")

while True:
    ret, frame = video.read()
    if not ret:
        break

    cv2.imshow("a", frame)
    if cv2.waitKey(1) == ord("q"):
        break
