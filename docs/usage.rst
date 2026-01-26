Usage
=====

The ``src`` directory contains code for model inference. The ``utils.py`` file
provides helper functions that cover most required functionalities.

Import scripts:

.. code-block:: py

   import sys
   sys.path.append("/path/to/src")
   from utils import ...


Examples
--------

For a full script, see
https://github.com/WaggleNet/BeeSee/blob/main/scripts/thorax_example.py


API
===

.. autofunction:: utils.load_dino_model

.. autofunction:: utils.preprocess_images

.. autofunction:: utils.extract_blobs

.. autoclass:: model_dino.DinoNN
