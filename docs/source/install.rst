.. _install:

Installation
============


**Cosmos** is a python program that will run in both Windows (we've tested it in Windows 10)
and Linux (we've tested it in Arch and Manjaro Distros).  

.. note::

    Practical use of Cosmos requires a computer with a CUDA-capable GPU.
    Follow NVIDIA CUDA Installation Guide for your OS:

    * `CUDA Installation Guide for Microsoft Windows <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_
    * `NVIDIA Arch Wiki <https://wiki.archlinux.org/index.php/NVIDIA>`_ and `cuda <https://www.archlinux.org/packages/community/x86_64/cuda/>`_:sup:`AUR`

.. note::

    We recommend to use Anaconda package manager to create virtual environment and install Cosmos in the virtual environment.
    After Anaconda installation create and activate a new environment in Anaconda Prompt:

    .. code-block:: bash

        conda create --name myenv python
        conda activate myenv

If you are using Anaconda package manager make sure that your created environment is activated (you should see the environment name (e.g., "myenv") in the command prompt.

Installation on Linux
~~~~~~~~~~~~~~~~~~~~~

Using pip::

    pip install git+https://github.com/gelles-brandeis/tapqir.git

From source::

    git clone https://github.com/gelles-brandeis/tapqir.git
    cd tapqir
    pip install .

Installation on Windows
~~~~~~~~~~~~~~~~~~~~~~~

Using pip::

    pip install git+https://github.com/gelles-brandeis/tapqir.git -f https://download.pytorch.org/whl/torch_stable.html

Updating Cosmos
~~~~~~~~~~~~~~~

To check tapqir version::

    tapqir --version

Update to latest verion of **tapqir**::

    pip install git+https://github.com/gelles-brandeis/tapqir.git -U

Set up Cosmos Server (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linux machines can be set up to run as servers. Following is a short instruction for Arch Linux.

Ssh server
----------

Install `OpenSSH <https://wiki.archlinux.org/index.php/OpenSSH#Installation>`_.
In ``/etc/ssh/sshd_config`` add the following line to allow access only for some users::

    AllowUsers    user1 user2

Change the default port from 22 to a random higher one like this::

    Port 39901

`Start/enable <https://wiki.archlinux.org/index.php/Systemd#Using_units>`_ ``sshd.service``.

Slurm server
------------

Follow instructions on `Slurm Arch Wiki <https://wiki.archlinux.org/index.php/Slurm>`_ and `Quick Start Administrator Guide <https://slurm.schedmd.com/quickstart_admin.html>`_. To create Slurm configuration file ``slurm.conf`` use the official `configurator <https://slurm.schedmd.com/configurator.easy.html>`_. Fill in the following options (same control and compute machines):

* *SlurmctldHost* - value returned by the :code:`hostname -s` in bash
* *Compute Machines* - values returned by the :code:`slurmd -C` command
* *StateSaveLocation* - change to ``/var/spool/slurm/slurmctld``
* *ProctrackType* - select ``LinuxProc``
* *ClusterName* - change to the same value as *SlurmctldHost*

Generate the file and copy it to ``/etc/slurm-llnl/slurm.conf``. Add following lines before COMPUTE NODES::

    # GENERAL RESOURCE
    GresType=gpu

Add ``Gres=gpu:x`` (``x`` is the number of gpu devices) to the NodeName line like this::

    NodeName=centaur Gres=gpu:2 CPUs=64 Sockets=1 CoresPerSocket=32 ThreadsPerCore=2 State=UNKNOWN RealMemory=64332

Finally, create ``/etc/slurm-llnl/gres.conf`` file by listing all gpu devices::

    #################################################################
    # Slurm's Generic Resource (GRES) configuration file
    ##################################################################
    # Configure support for our four GPUs
    Name=gpu File=/dev/nvidia0 CPUs=0-4
    Name=gpu File=/dev/nvidia1 CPUs=5-9

`Start/enable <https://wiki.archlinux.org/index.php/Systemd#Using_units>`_ ``slurmd.service`` and ``slurmctld.service``.


Remote Desktop Server
------------------------

Install `xrdp <https://wiki.archlinux.org/index.php/Xrdp>`_ package on the Linux server machine.
`Start/enable <https://wiki.archlinux.org/index.php/Systemd#Using_units>`_ ``xrdp.service`` and ``xrdp-sesman.service``.

.. note::

    Connect from the University network or use VPN client.
    Use remote desktop program (`Remmina <https://wiki.archlinux.org/index.php/Remmina>`_ on Linux) to connect to the computer.
    At the login screen select xvnc display session.
