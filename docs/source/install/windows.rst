Install on Windows
==================

First, make sure that `CUDA is installed`_ on the computer.

.. important::

   We strongly recommend creating a virtual environment to encapsulate your
   installation. Below we use a popular package manager Anaconda to create
   and manage a virtual environment.

1. `Install Anaconda`_ package manager.

2. Open an Anaconda Prompt. Create a new environment and give it a name 
   (e.g., ``tapqir-env``)::

    > conda create --name tapqir-env python=3.8

3. Activate the environement (you should see the environment name
   (i.e., ``tapqir-env``) in the command prompt)::

    > conda activate tapqir-env

4. Install `git <https://git-scm.com/>`_ (this is needed because we will
   install ``tapqir`` from the GitHub repository)::

    > conda install -c anaconda git

5. To install ``tapqir``, in the Anaconda Prompt, run::

    > pip install git+https://github.com/gelles-brandeis/tapqir.git

.. _CUDA is installed: https://developer.nvidia.com/cuda-downloads
.. _Install Anaconda: https://docs.anaconda.com/anaconda/install/
