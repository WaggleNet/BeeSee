## U-Net bee detection

The Reduced U-Net from the [2017 OIST paper](https://arxiv.org/pdf/1712.08324) is implemented.

In this implementation, the network detects bees in a dense context.

We omit the recurrent neural network, so bee tracking is likely not ideal.

Usage:

Download the frames of 30FPS dataset at https://groups.oist.jp/bptu/honeybee-tracking-dataset

Run `python dataset.py --data /path/to/dataset/` to preview data (saves to "x.png", "y.png").

Run `python train.py --data /path/to/dataset/` to train the model.
