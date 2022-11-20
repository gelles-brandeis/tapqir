Install on Windows 11
=====================

.. note::

   Tapqir does not work natively on Windows but can be run in a Windows Subsystem for Linux 2 (WSL 2) window in Windows 11.  (As far as we know, this does *not* work in Windows 10 or earlier versions.)

1. Install *Windows* Nvidia drivers if not already installed.

   `Link to Nvidia drivers <https://www.nvidia.com/download/index.aspx>`_

2. Install WSL 2 (Ubuntu).

   Install "Ubuntu" using the Windows Store app.  Run Ubuntu and in the terminal do::
   
    $ sudo apt update

   .. code-block::

    $ sudo apt upgrade

3. Install g++ (>=7) if not already installed.
   
   To check installation versions in the WSL terminal run::

    $ g++ --version

   To install (if not already installed) in the terminal run::

    $ sudo apt install g++
    
4. Install latest version of CUDA (needs to be version 11.5 or later).

   Summary of `CUDA installation instructions <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl-installation>`_::

    $ sudo apt-key del 7fa2af80

   .. code-block::

    $ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb

   .. code-block::

    $ sudo dpkg -i cuda-keyring_1.0-1_all.deb

   .. code-block::

    $ sudo apt update

   .. code-block::

    $ sudo apt install cuda

5. Install the linux version of the Anaconda package manager (`Anaconda installation instructions <https://docs.anaconda.com/anaconda/install/linux/>`_).
   Here is the summary of required installation steps:

   * Download installer from `<https://www.anaconda.com/products/individual>`_ (anaconda nucleus sign-up page can be ignored).

   * Navigate to the directory containing the installer.  If you downloaded it using a Windows web browser, this will be in ``/mnt/c/Users/<your Windows username>/Downloads``.
   
   * Run the following command to install Anaconda (change the name of the installer file appropriately if it
     is a newer version)::

      $ bash Anaconda3-2021.11-Linux-x86_64.sh
    
   * Press Enter at the “In order to continue the installation process, please review the license agreement.” prompt.
   
   * Scroll to the bottom of the license terms and enter “Yes” to agree.
   
   * Press Enter to accept the default install location.
   
   * Type "yes" at “Do you wish the installer to initialize Anaconda3 by running conda init?” prompt.
   
   * After installation is complete *close the terminal and open it again*. Now you should see ``(base)`` environment indicated in the terminal.

6. Create a new environment and give it a name (e.g., ``tapqir-env``)::

    $ conda create --name tapqir-env python=3.8

7. Activate the environement (you should see the environment name
   (i.e., ``tapqir-env``) in the command prompt)::

    $ conda activate tapqir-env

8. Install ``tapqir``::

    $ pip install tapqir[desktop]

Now you can run Tapqir in the WSL window in the same way you would on a linux computer.

Update Tapqir
-------------

To update Tapqir run (make sure that ``tapqir-env`` environment is activated)::

   $ pip install tapqir -U

Check version
-------------

To check Tapqir version run::

   $ tapqir --version

Tapqir on WSL Usage tips
------------------------

When working with Tapqir in WSL, it is most convenient to work in subdirectories of the linux directory /home/<your linux username>, which is the same as the Windows directory ``\\\\wsl.localhost\\Ubuntu\\home\\<your linux username>``.

When running tapqir-gui, browser windows will not open automatically.  Look for a message like::

     [Voila] Voilà is running at: http://localhost:8866/
     
in the console window and open that URL in a windows web browsser to access the GUI.

.. tip::

   If there are two GPUs on your computer, use::

      $ CUDA_VISIBLE_DEVICES=0 tapqir-gui

   to run Tapqir on the first GPU, or::     
   
      $ CUDA_VISIBLE_DEVICES=1 tapqir-gui

   to run Tapqir on the second GPU.  You can use these two commends in two different Ubuntu windows, so that you can process data in twwo separate Tapqir runs at the same time.
