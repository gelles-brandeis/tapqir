.. _usage:

User Guide
==========

.. note::

    Checkout a :doc:`tutorial </tutorials/tutorial_part_i>` analysis of a sample data set that
    can be followed along with this user guide or used as a template for your own analysis.

Set up the environment
----------------------

Windows/Linux
^^^^^^^^^^^^^

1. If Tapqir is not installed, please follow these :doc:`instructions </install/index>` to do so.

2. Running Tapqir requires command-line interface. On Windows open the Anaconda Prompt.
   On Linux open the terminal.

3. Activate the virtual environment (e.g., if named ``tapqir-env``)::

   $ conda activate tapqir-env

Google Colab
^^^^^^^^^^^^

To get started in Google Colab follow :doc:`the installation guide </install/colab>`.

Create & initialize Tapqir analysis folder
------------------------------------------

To start the analysis create an empty folder and initialize it by running
``tapqir init`` inside the new folder::

    $ mkdir new_folder
    $ cd new_folder
    $ tapqir init

``tapqir init`` command will create a ``.tapqir`` folder that will store internal files
such as ``config`` file, ``log`` file, and model checkpoints.

Preprocessing raw input data
----------------------------

Data from glimpse/imscroll
^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyzing data acquired with `Glimpse <https://github.com/gelles-brandeis/Glimpse>`_ and pre-processed with 
the `imscroll <https://github.com/gelles-brandeis/CoSMoS_Analysis/wiki>`_ program
will require the following files:

* image data in glimpse format and header file
* aoiinfo file designating the locations of target molecules (on-target AOIs) in the binder channel
* (optional) aoiinfo file designating the off-target control locations (off-target AOIs) in the binder channel
* driftlist file recording the stage movement that took place during the experiment

To extract AOIs from raw images run::

    $ tapqir glimpse --title <name> --header-dir <path> --aoi-size <number> --ontarget-aoiinfo <path> --offtarget-aoiinfo <path> --driftlist <path> --frame-start <number> --frame-end <number>

Options:

* ``--title`` - Project/experiment name

* ``--aoi-size`` - AOI image size - number of pixels along the axis (default: 14)

* ``--header-dir`` - Path to the header/glimpse folder

* ``--ontarget-aoiinfo`` - Path to the on-target AOI locations file

* ``--offtarget-aoiinfo`` - Path to the off-target control AOI locations file (optional)

* ``--driftlist`` - Path to the driftlist file

* ``--frame-start`` - First frame to include in the analysis (optional)

* ``--frame-end`` - Last frame to include in the analysis (optional)

Optionally starting and ending frames can be specified by ``--frame-start`` and
``--frame-end``, otherwise, full range of frames from the driftlist file will be analyzed.

The program will output ``data.tpqr`` file containing extracted AOIs, target
(and off-target control) locations, empirical offset distirbution.

Data analysis
-------------

Fit the data to the time-independent ``cosmos`` model with :math:`\theta`
marginalized out (``--marginal``)::

    $ tapqir fit cosmos --marginal --cuda --bs <number> --num-iter <number>

Options:

* ``--cuda`` - Run computations on GPU.

* ``--bs <number>`` - Batch size is the size of the subset of AOIs that is used
  for fitting at each iteration. It affects the amount of memory consumed and
  computation time and can be adjusted accordingly. ``nvidia-smi`` shell command shows
  Memory-Usage and GPU-Util values. Our recommendation is to increase batch size till
  GPU-Util is high (> 60%) but not maxed out.

* ``--num-iter <number>`` - Number of fitting iterations. Setting it to 0 will run the program till 
  Tapqir's custom convergence criteria is satisfied. We recommend to set it to 0 (default)
  and then run for additional number of iterations as required. Convergence of global
  parameters can be visually checked using tensorboard_.

The program will save a checkpoint every 100 iterations (checkpoint is saved at ``.tapqir/cosmos-model.tpqr``).
Starting the program again will resume from the last saved checkpoint. At every checkpoint the values of global
variational parameters (``-ELBO``, ``gain_loc``, ``proximity_loc``, ``pi_mean``, ``lamda_loc``) are also
recorded for visualization by tensorboard_. Plateaued plots signify convergence.

After the marginalized (``--marginal``) model has converged run the full ``cosmos`` model (usually
15,000-20,000 iterations is enough)::

    $ tapqir fit cosmos --cuda --bs <number> --num-iter <number>

.. tip::

    Use ``CUDA_VISIBLE_DEVICES`` environment variable to change CUDA device::

        $ CUDA_VISIBLE_DEVICES=1 tapqir fit ...

    To view available devices run::

        $ nvidia-smi

Tensorboard
^^^^^^^^^^^

Fitting progress can be inspected while fitting is taking place or afterwards using `tensorboard program <https://www.tensorflow.org/tensorboard>`_::

    $ tensorboard --logdir=.

Posterior distributions
^^^^^^^^^^^^^^^^^^^^^^^

To compute 95% credible intervals of model parameters run::

    $ tapqir stats cosmos --matlab

Options:

* ``--matlab`` - Save parameters in matlab format (default: False)

Parameters with their mean value, 95% CI (credible interval) lower limit and upper limit
are saved in ``cosmos-params.tqpr``, ``cosmos-params.mat``, and ``cosmos-summary.csv`` files.

To visualize analysis results run::

    $ tapqir show cosmos

which will open GUI displaying parameter values (mean and 95% CI). Clicking on the ``Images`` button
will show original images along with the best fit estimates.

Viewing logging info
--------------------

Tapqir logs console output to a ``.tapqir/loginfo`` text file. It can be viewed by running::

    $ tapqir log

..
    Configuration file
    ~~~~~~~~~~~~~~~~~~

    Tapqir stores command options in the configuration file ``.tapqir/config``. When the program is run
    command option values are automatically saved in the ``config`` file and used as a default value in
    the next invocation. To manually change option values ``tapqir config`` command can be used::

        $ tapqir config <name> <value>

    where

    * ``<value>`` - Option name (command.option). For example ``fit.bs``

Using Slurm
-----------

If `Slurm Workload Manager <https://slurm.schedmd.com/documentation.html>`_ is
configured on the machine Tapqir analysis can be submitted as a slurm job::

    $ sbatch --job-name <name> --gres gpu:1 tapqir fit <model> --cuda --bs <number> --num-iter <number>

Sbatch command options:

* ``--job-name`` - Job name.
* ``--gres`` - Generic Resources (``<type>:<amount>``).
