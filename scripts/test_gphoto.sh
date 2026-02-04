#!/bin/bash

#gphoto2 --capture-image-and-download

gphoto2 --stdout --capture-movie | ffplay -
