Install on Linux
================

We have tested Tapqir installation on Ubuntu 20.04 and Arch Linux distributions.

1. Install Nvidia drivers if not already installed.

   *On Ubuntu 20.04*
   
   .. tip::

      On Ubuntu select the text to copy it and then press the middle mouse button (scrolling wheel) to paste the copied text.
   
   To get information about your graphic card and available drivers run::

    $ ubuntu-drivers devices
    
    // From the output
    vendor   : NVIDIA Corporation
    model    : TU102 [GeForce RTX 2080 Ti]
    driver   : nvidia-driver-470 - distro non-free recommended

   Install the recommended nvidia driver (in this case ``nvidia-driver-470``)::

    $ sudo apt install nvidia-driver-470

   *On Arch Linux*

   Install the nvidia package::

    $ sudo pacman -S nvidia

2. Install g++ (>=7) if not already installed.
   
   To check installation versions in the terminal run::

    $ g++ --version

   To install (if not already installed):

   *On Ubuntu 20.04*

   In the terminal run::

    $ sudo apt install g++

   *On Arch Linux*

   In the terminal run::

    $ sudo pacman -S gcc
    
4. Install latest version of CUDA (needs to be version 11.5 or later).

   *On Ubuntu 20.04*

   Summary of `CUDA installation instructions <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation>`_::

    $ sudo apt-key del 7fa2af80

   .. code-block::

    $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb

   .. code-block::

    $ sudo dpkg -i cuda-keyring_1.0-1_all.deb

   .. code-block::

    $ sudo apt-get update

   .. code-block::

    $ sudo apt-get install cuda

   Reboot the system::

    $ sudo reboot

   *On Arch Linux*

   In the terminal run::

    $ sudo pacman -S cuda

   Reboot the system::

    $ sudo reboot

5. Install Anaconda package manager (`Anaconda installation instructions <https://docs.anaconda.com/anaconda/install/linux/>`_).
   Here is the summary of required installation steps:

   * Download installer from `<https://www.anaconda.com/products/individual>`_ (anaconda nucleus sign-up page can be ignored).

   * Run the following command to install Anaconda (change the name of the installer file appropriately if it
     is a newer version)::

      $ bash ~/Downloads/Anaconda3-2021.11-Linux-x86_64.sh
    
   * Press Enter at the “In order to continue the installation process, please review the license agreement.” prompt.
   
   * Scroll to the bottom of the license terms and enter “Yes” to agree.
   
   * Press Enter to accept the default install location.
   
   * Type "yes" at “Do you wish the installer to initialize Anaconda3 by running conda init?” prompt.
   
   * After installation is complete close the terminal and open it again. Now you should see ``(base)`` environment indicated in the terminal.

6. Create a new environment and give it a name (e.g., ``tapqir-env``)::

    $ conda create --name tapqir-env python=3.8

7. Activate the environement (you should see the environment name
   (i.e., ``tapqir-env``) in the command prompt)::

    $ conda activate tapqir-env

8. Install ``tapqir``::

    $ pip install tapqir[desktop]

.. tip::

   If there are two GPUs on your computer, use::

      $ CUDA_VISIBLE_DEVICES=1 tapqir-gui

   to run Tapqir on the second GPU.

Update Tapqir
-------------

To update Tapqir run (make sure that ``tapqir-env`` environment is activated)::

   $ pip install tapqir -U

Check version
-------------

To check Tapqir version run::

   $ tapqir --version

Install linux server tools (optional)
-------------------------------------

Linux machines can be set up to run as servers for batch processing of Tapqir runs. This is optional
and requires some linux sysadmin skills.  The following are short summary instructions for installing the server 
tools on Arch Linux.

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

    Use remote desktop program (`Remmina <https://wiki.archlinux.org/index.php/Remmina>`_ on Linux) to connect to the computer.
    At the login screen select xvnc display session.
