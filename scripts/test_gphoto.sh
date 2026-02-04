#!/bin/bash

# This script sets up a v4l2 loopback /dev/video11,
# and has gphoto2/ffmpeg stream video into it.
# Then, a Python script can stream from that.

sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=11 card_label=GPhoto exclusive_caps=1

gphoto2 --stdout --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -f v4l2 /dev/video11

#gphoto2 --capture-image-and-download
#gphoto2 --stdout --capture-movie | ffplay -
