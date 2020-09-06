************
Installation
************


**cosmos** is a python program that will run in both Windows (we've tested it in Windows 10) and linux (we've tested it in Arch linux).  

Practical use requires a computer with an NVIDIA GPU capable of running **cuda**.

Follow these steps to install the software: 

Install Nvidia drivers and cuda
-------------------------------

**Linux**

First install nvidia drivers (`Arch linux <https://wiki.archlinux.org/index.php/NVIDIA#Installation>`_ or `Manjaro <https://wiki.manjaro.org/index.php?title=Configure_NVIDIA_(non-free)_settings_and_load_them_on_Startup#Install_NVIDIA_Drivers>`_) and then install cuda:

.. code-block:: bash

    sudo pacman -S cuda

**Windows**

Follow nvidia/cuda installation `instructions <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_.

Create virtual environment
--------------------------

**Linux**

Install Anaconda from Arch linux aur `repository <https://aur.archlinux.org/packages/anaconda/>`_:

.. code-block:: bash

    git clone https://aur.archlinux.org/anaconda.git
    cd anaconda
    makepkg -si
    source /opt/anaconda/bin/activate root  # activates conda (do for each user)
    conda init # adds autostart script to your ~/.bashrc (do for each user)
