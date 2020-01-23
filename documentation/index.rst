.. 2d-imaging documentation master file, created by
   sphinx-quickstart on Tue Oct 30 21:00:24 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to 2d-imaging!
======================================
.. image:: stanford_ml_group.png

2d-imaging allows training and testing classification models for 2d images.

The source code for 2d-imaging can be `found here <https://github.com/stanfordmlgroup/aihc-fall18-2dimaging>`_.

Getting Started
---------------

1. Activate environment: `source activate chxr`
2. Train a model: run `python train.py -h` for information about available options.
3. Open TensorBoard:
    - While training, launch TensorBoard: `tensorboard --logdir=logs --port=5678`
    - Port forward: `ssh -N -f -L localhost:1234:localhost:5678 <SUNET>@bootcamp`
    - View in browser: `http://localhost:1234/`


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   train
   test
   dataset
   models


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
