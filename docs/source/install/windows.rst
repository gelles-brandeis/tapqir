Install on Windows
==================

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

4. Install CUDA and ensure that it is version 11.5 or later::

    > conda install cuda -c nvidia
    > nvcc --version

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2021 NVIDIA Corporation
    Built on Mon_Sep_13_19:13:29_PDT_2021
    Cuda compilation tools, release 11.5, V11.5.50
    Build cuda_11.5.r11.5/compiler.30411180_0

5. Install `git <https://git-scm.com/>`_ (this is needed because we will
   install ``tapqir`` from the GitHub repository)::

    > conda install -c anaconda git

6. To install ``tapqir``, in the Anaconda Prompt, run::

    > pip install git+https://github.com/gelles-brandeis/tapqir.git -f https://download.pytorch.org/whl/torch_stable.html

.. _Install Anaconda: https://docs.anaconda.com/anaconda/install/
