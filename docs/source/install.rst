.. _install:

Installation
============


**cosmos** is a python program that will run in both Windows (we've tested it in Windows 10) and linux (we've tested it in Arch linux).  

Practical use requires a computer with an NVIDIA GPU capable of running **cuda**.

Follow these steps to install the software: 

Install Nvidia drivers and cuda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux**

First install nvidia drivers (`Arch linux <https://wiki.archlinux.org/index.php/NVIDIA#Installation>`_ or `Manjaro <https://wiki.manjaro.org/index.php?title=Configure_NVIDIA_(non-free)_settings_and_load_them_on_Startup#Install_NVIDIA_Drivers>`_) and then install cuda:

.. code-block:: bash

    sudo pacman -S cuda

**Windows**

Follow nvidia/cuda installation `instructions <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_.

Create virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux**

Install Anaconda from Arch linux aur `repository <https://aur.archlinux.org/packages/anaconda/>`_:

.. code-block:: bash

    git clone https://aur.archlinux.org/anaconda.git
    cd anaconda
    makepkg -si
    source /opt/anaconda/bin/activate root  # activates conda (do for each user)
    conda init # adds autostart script to your ~/.bashrc (do for each user)

**Windows**

Install Anaconda for `Windows <https://docs.anaconda.com/anaconda/install/>`_.

**Both OSs**

After installation create and activate a new environment in anaconda prompt:

.. code-block:: bash

    conda create --name myenv python=3.7
    conda activate myenv

Install cosmos
~~~~~~~~~~~~~~

First, make sure that your created environment is activated (you should see the environment name (e.g., "myenv") in the command prompt. Then download and install the **cosmos** software:

**Linux**

.. code-block:: bash

    git clone https://github.com/gelles-brandeis/cosmos.git
    cd cosmos
    pip install .

**Windows**

.. code-block:: bash

    git clone https://github.com/gelles-brandeis/cosmos.git
    cd cosmos
    pip install . -f https://download.pytorch.org/whl/torch_stable.html

Once **cosmos** has been installed:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update to latest verion of **cosmos**
-------------------------------------

Go to cosmos repository and then run:

.. code-block:: bash

    git pull
    pip install . -U

Check cosmos version:
---------------------

.. code-block:: bash

    cosmos --version

Setting up **cosmos** Server (Arch Linux)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ssh server
----------

**Install openssh**

.. code-block:: bash

    sudo pacman -S openssh

**Configuration**

In `/etc/ssh/sshd_config` add the following line to allow access only for some users:

.. code-block:: bash

    AllowUsers    user1 user2

Change the default port from 22 to a random higher one like this:

.. code-block:: bash

    Port 39901

**Start/enable daemon**

.. code-block:: bash

    sudo systemctl start sshd
    sudo systemctl enable sshd

Slurm server
------------

Follow instructions on `arch wiki <https://wiki.archlinux.org/index.php/Slurm) and slurm [administrator guide](https://slurm.schedmd.com/quickstart_admin.html>`_.

Briefly, install `slurm-llnl` and `munge` packages. Create `slurm.conf` file using the official `configurator <https://slurm.schedmd.com/configurator.easy.html>`_. Fill in the following options (same control and compute machines):
* **SlurmctldHost** - value returned by the `hostname -s` command in bash
* **Compute Machines** - values returned by the `slurmd -C` command
* **StateSaveLocation** - change to `/var/spool/slurm/slurmctld`
* **ProctrackType** - select _LinuxProc_
* **ClusterName** - change to the same value as **SlurmctldHost**

Generate the file and copy it to `/etc/slurm-llnl/slurm.conf`. Add following lines before COMPUTE NODES:

.. code-block:: bash

    # GENERAL RESOURCE
    GresType=gpu

Add `Gres=gpu:x` (_x_ is the number of gpu devices) to the NodeName line like this:

.. code-block:: bash

    NodeName=centaur Gres=gpu:2 CPUs=64 Sockets=1 CoresPerSocket=32 ThreadsPerCore=2 State=UNKNOWN RealMemory=64332

Finally, create `/etc/slurm-llnl/gres.conf` file by listing all gpu devices:

.. code-block:: bash

    #################################################################
    # Slurm's Generic Resource (GRES) configuration file
    ##################################################################
    # Configure support for our four GPUs
    Name=gpu File=/dev/nvidia0 CPUs=0-4
    Name=gpu File=/dev/nvidia1 CPUs=5-9

Start/enable `slurmd` and `slurmctld` services


Xrdp server (Remote Desktop for Linux)
--------------------------------------

**Server setup**

Install `xrdp <https://wiki.archlinux.org/index.php/Xrdp>`_ package from the aur.

Start/enable xrdp and xrdp-sesman services

.. code-block:: bash

    sudo systemctl start xrdp
    sudo systemctl start xrdp-sesman
    sudo systemctl enable xrdp
    sudo systemctl enable xrdp-sesman

**Client side**

Connect from the University network or use VPN client. Use remote desktop program (Remmina on Linux) to connect to the computer.

At the login screen select xvnc display session.
