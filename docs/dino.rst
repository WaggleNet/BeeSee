DINO
====

Overview
--------

DINO is a research project and neural network developed by Facebook AI.

DINO is a Vision Transformer. Thus, the input is an image.
The output is a vector in latent space.

During training, the model is given a large amount of unlabeled images.
The images are put through various transforms (crop, rotate, scale, etc.).
The cost function is to learn the same latent output for the same image,
even when transformed.
This results in the model learning visually salient features, but not for
any specific task.

The intermediate layers provide useful information. They are a sequence
of tokens (this is a transformer), where each token corresponds to a 14x14
patch of pixels (this is a vision transformer).

Thus, the intermediate layers provide a patch-level latent representation.
I.e. a latent representation of the image, scaled down 14x resolution.

We train a very small (1K params) convolutional head, which takes in
the DINO intermediate layer. We train it for a specific task, e.g. segment
the thorax.
