Installation
============


Tapqir is a Python program and can be installed via ``pip`` package manager.

.. important::

   Practical use of Tapqir requires a computer with a CUDA-capable GPU. If you are
   using a Linux or a Windows machine first `install CUDA`_. Alternatively, 
   Tapqir can be run in `Colab notebooks`_, Google's cloud servers that provide
   free access to GPUs.

Linux
-----

.. important::

   We strongly recommend creating a virtual environment to encapsulate your
   installation. For example use Anaconda_ package manager to create
   a new environment::

        $ conda create --name tapqir-env python=3.8
        $ conda activate tapqir-env

   Before installing & using Tapqir make sure that your created environment
   is activated (you should see the environment name (e.g., ``tapqir-env``)
   in the command prompt).

.. _install CUDA: https://developer.nvidia.com/cuda-downloads
.. _Colab notebooks: https://colab.research.google.com/notebooks/intro.ipynb
.. _Anaconda: https://docs.anaconda.com/anaconda/install/

In the terminal run::

    $ pip install git+https://github.com/gelles-brandeis/tapqir.git

Google Colab
------------

Colab notebooks start in a fresh environment and thus require installation for each new
start. Before installing ``tapqir``, switch runtime to GPU (in menu select ``Runtime ->
Change runtime type -> GPU``) and mount Google Drive to save analysis output. Install
``tapqir`` using pip (installation output is silenced)::

    !pip install --quiet git+https://github.com/gelles-brandeis/tapqir.git > install.log

Development
-----------

For development - install from source::

    $ git clone https://github.com/gelles-brandeis/tapqir.git
    $ cd tapqir
    $ make install

Set up a slurm server (optional)
--------------------------------

Linux machines can be set up to run as servers. Following is a short instruction
for Arch Linux.

ssh server
~~~~~~~~~~

Install `OpenSSH <https://wiki.archlinux.org/index.php/OpenSSH#Installation>`_.
In ``/etc/ssh/sshd_config`` add the following line to allow access only for some users::

    AllowUsers    user1 user2

Change the default port from 22 to a random higher one like this::

    Port 39901

`Start/enable <https://wiki.archlinux.org/index.php/Systemd#Using_units>`_ ``sshd.service``.

slurm server
~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~

Install `xrdp <https://wiki.archlinux.org/index.php/Xrdp>`_ package on the Linux server machine.
`Start/enable <https://wiki.archlinux.org/index.php/Systemd#Using_units>`_ ``xrdp.service`` and ``xrdp-sesman.service``.

.. note::

    Connect from the University network or use VPN client.
    Use remote desktop program (`Remmina <https://wiki.archlinux.org/index.php/Remmina>`_ on Linux) to connect to the computer.
    At the login screen select xvnc display session.
