User Guide
==========

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin the analysis first create an empty directory and run::

    tapqir config path_to_folder

where ``path_to_folder`` is the newly created folder. This command
will create a text file named ``options.cfg`` containing command options.

Data preparation
~~~~~~~~~~~~~~~~

Data from glimpse/imscroll
--------------------------

Analyzing data acquired with `Glimpse <https://github.com/gelles-brandeis/Glimpse>`_ and pre-processed with 
the ``imscroll`` program (see `CoSMoS_Analysis <https://github.com/gelles-brandeis/CoSMoS_Analysis/wiki>`_)
will require the following files:

- image data folder in glimpse format
- on-target ``aoiinfo`` file designating AOIs corresponfing to target molecules to be analyzed
- (optional) off-target ``aoiinfo`` file designating AOIs corresponding to locations that do not contain target molecules
- ``driftlist`` file recording the stage movement that took place during the experiment

Enter the names of your folder/files under the ``[glimpse]`` section of the ``options.cfg`` file::

    [glimpse]
    dir = /path/to/glimpse/folder
    ontarget_aoiinfo = /path/to/ontarget_aoiinfo_file
    offtarget_aoiinfo = /path/to/offtarget_aoiinfo_file
    driftlist = /path/to/driftlist_file
    frame_start
    frame_end
    ontarget_labels
    offtarget_labels
    
(``frame_start``, ``frame_end``, ``ontarget_labels`` and ``offtarget_labels`` are optional)

To process your data run::

    tapqir glimpse path_to_folder
    
The program will save ``data.tpqr`` file containing the digested data in the format needed for fitting.

Fit the data to a model
~~~~~~~~~~~~~~~~~~~~~~~

To fit the data run::

    tapqir fit cosmos path_to_folder

where `cosmos` is the name of the model.

If necessary, fitting parameters can be edited in the ``[fit]`` section of the ``options.cfg`` file::

    [fit]
    num_states = 1
    k_max = 2
    num_iter = 60000
    num_samples = 1000
    batch_size = 0
    learning_rate = 0.005
    control = True
    device = cuda
    dtype = double
    jit = False
    backend = pyro

.. note::

    To use different CUDA device run::

        CUDA_VISIBLE_DEVICES=1 tapqir fit cosmos path_to_folder

Tensorboard
-----------

Fitting progress can be visualized using tensorboard program::

    tensorboard --logdir=path_to_folder

which will open the window in the browser:

.. image:: tensorboard.png

Posterior Distributions
~~~~~~~~~~~~~~~~~~~~~~~

Compute posterior distributions of model parameters with ``save`` command::

    tapqir save cosmos path_to_folder

Posterior distributions of global parameters are saved in ``statistics.csv`` file.

Posterior distributions of local parameters are saved in ``params.tpqr`` and
can be visualized by running ``show`` command::

    tapqir show cosmos path_to_folder

which will display parameter values, original images along with best estimates:

.. image:: parameters.png

.. image:: images.png
