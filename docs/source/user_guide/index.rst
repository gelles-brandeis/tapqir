.. _usage:

User Guide
==========

Initialize Tapqir analysis folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start the analysis with Tapqir create an empty folder and initialize it
by running  ``tapqir init`` inside the new folder::

    $ mkdir new_folder
    $ cd new_folder
    $ tapqir init

``tapqir init`` command will create a ``.tapqir`` folder that will store internal files
such as ``config`` file and ``log`` file.

Preprocessing raw input data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data from glimpse/imscroll
--------------------------

Analyzing data acquired with `Glimpse <https://github.com/gelles-brandeis/Glimpse>`_ and pre-processed with 
the `imscroll <https://github.com/gelles-brandeis/CoSMoS_Analysis/wiki>`_ program
will require the following files:

* image data in glimpse format and header file
* aoiinfo file designating the locations of target molecules (on-target AOIs) in the binder channel
* (optional) aoiinfo file designating the off-target control locations (off-target AOIs) in the binder channel
* driftlist file recording the stage movement that took place during the experiment

To extract AOIs from raw images run::

    $ tapqir glimpse --title <name> --header-dir <path> --ontarget-aoiinfo <path> --offtarget-aoiinfo <path> --driftlist <path> --frame-start <number> --frame-end <number>

Options:

* ``--title`` - Project name

* ``--header-dir`` - Path to the header/glimpse folder

* ``--ontarget-aoiinfo`` - Path to the on-target AOI locations file

* ``--offtarget-aoiinfo`` - Path to the off-target control AOI locations file (optional)

* ``--driftlist`` - Path to the driftlist file

* ``--frame-start`` - First frame to include in the analysis (optional)

* ``--frame-end`` - Last frame to include in the analysis (optional)

Optionally starting and ending frames can be specified by ``frame_start`` and ``frame_end``,
otherwise, full range of frames from the driftlist file will be analyzed.

The program will output ``data.tpqr`` file containing extracted AOIs, target
(and off-target control) locations, empirical offset distirbution.

Data analysis
~~~~~~~~~~~~~

Fit the data to the ``marginal`` model (time-independent model with :math:`\theta`
marginalized out)::

    $ tapqir fit marginal --cuda -bs <number> -it <number>

Options:

* ``--cuda`` - Run computations on GPU

* ``-bs <number>`` - Batch size is the size of the subset of AOIs that is used
  for fitting at each iteration. It affects the amount of memory consumed and
  computation time and can be adjusted accordingly. ``nvidia-smi`` command shows
  Memory-Usage and GPU-Util values. Our recommendation is to increase batch size till
  GPU-Util is high (> 70%) but not maxed out.

* ``-it <number>`` - Number of fitting iterations. Setting it to 0 will run the program
  till our custom convergence criteria is satisfied. We recommend to set it to 0 (default)
  and then run for additional number of iterations if necessary. Convergence of global
  parameters can be visually checked using `tensorboard`_.

After ``marginal`` model has converged run the full ``cosmos`` model (usually
15000-20000 iterations is enough)::

    $ tapqir fit cosmos --cuda -bs <number> -it <number>

.. tip::

    Use ``CUDA_VISIBLE_DEVICES`` environment variable to change CUDA device::

        $ CUDA_VISIBLE_DEVICES=1 tapqir fit ...

Tensorboard
-----------

Fitting progress can be visualized using tensorboard_ program::

    tensorboard --logdir=.

.. _tensorboard: https://www.tensorflow.org/tensorboard

Posterior distributions
-----------------------

To compute 95% credible intervals of model parameters run::

    $ tapqir stats cosmos --matlab

Options:

* ``--matlab`` - Save parameters in matlab format (default: False)

Parameters with their mean value, 95% CI lower limit and upper limit are saved in
``cosmos-params.tqpr``, ``params.mat``, and ``statistics.csv`` files.

To visualize analysis results run::

    $ tapqir show cosmos

which will display parameter values (mean and 95% CI), original images along with
the best fit estimates.

Using slurm
~~~~~~~~~~~

If `Slurm Workload Manager <https://slurm.schedmd.com/documentation.html>`_ is
configured on the machine Tapqir analysis can be submitted as a slurm job::

    $ sbatch --job-name <name> --gres gpu:1 tapqir fit <model> --cuda -bs <number> -it <number>

What's next?
~~~~~~~~~~~~

Checkout our google colab tutorial:
