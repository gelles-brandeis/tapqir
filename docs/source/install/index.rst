Installation
============


Tapqir is a Python program that can be installed via ``pip`` package manager.

.. important::

   Practical use of Tapqir requires a computer with a CUDA-capable GPU. If you are
   using a Linux or a Windows machine first `install CUDA`_. Alternatively, 
   Tapqir can be run in `Colab notebooks`_, Google's cloud servers that provide
   free access to GPUs.

.. _install CUDA: https://developer.nvidia.com/cuda-downloads
.. _Colab notebooks: https://colab.research.google.com/notebooks/intro.ipynb

.. toctree::
   :maxdepth: 1

   windows
   linux
   colab

Advanced options
----------------

- On Linux machines you can :doc:`set up a Slurm server <linux>` to be able to
  submit and queue Tapqir jobs.
