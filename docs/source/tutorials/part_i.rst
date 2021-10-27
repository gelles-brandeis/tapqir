Part I: Intro
=============

Models
------

Tapqir's most basic model, ``cosmos``, was developed for colocalization detection in a
relatively simple binder-target single-molecule experiment. ``cosmos`` model is
*time-independent* and *single-color*. Please checkout our preprint (`Ordabayev et al., 2021`_)
for a comprehensive description of the ``cosmos`` model. Tapqir can be extended readily
to more complex models. For example, we expect Tapqir to naturally extend to multi-state,
multi-color, and kinetic analysis.

``cosmos`` is a *Bayesian* model, i.e. each model parameter has an associated probability
distribution (uncertainty). For those who are interested, `Kinz-Thompson et al., 2021`_ is
a nice read about Bayesian inference in the context of single-molecule data analysis.

As a consequence of Bayesian inference, Tapqir computes for each frame of each AOI the *probability*
:math:`p(\mathsf{specific})`, that a target-specific spot is present.

``cosmos`` is a physics-informed model, i.e. model parameters have a physical interpretation.
For *N* AOIs, *F* frames, and *K* spots model parameters are:

+------------------------+-----------+-------------------------------------+
| Parameter              | Shape     | Description                         |
+========================+===========+=====================================+
| ``gain`` - :math:`g`   | (1,)      | camera gain                         |
+------------------------+-----------+-------------------------------------+
| ``proximity`` - |prox| | (1,)      | proximity                           |
+------------------------+-----------+-------------------------------------+
| ``lamda`` - |ld|       | (1,)      | average rate of target-nonspecific  |
|                        |           | binding                             |
+------------------------+-----------+-------------------------------------+
| ``pi`` - :math:`\pi`   | (1,)      | average binding probability of      |
|                        |           | target-specific binding             |
+------------------------+-----------+-------------------------------------+
| ``theta`` - |theta|    | (N, F)    | target-specific spot index          |
+------------------------+-----------+-------------------------------------+
| ``m`` - :math:`m`      | (K, N, F) | spot presence indicator             |
+------------------------+-----------+-------------------------------------+
| ``height`` - :math:`h` | (K, N, F) | spot intensity                      |
+------------------------+-----------+-------------------------------------+
| ``width`` - :math:`w`  | (K, N, F) | spot width                          |
+------------------------+-----------+-------------------------------------+
| ``x`` - :math:`x`      | (K, N, F) | spot position on x-axis             |
+------------------------+-----------+-------------------------------------+
| ``y`` - :math:`y`      | (K, N, F) | spot position on y-axis             |
+------------------------+-----------+-------------------------------------+
| ``background`` - |b|   | (N, F)    | background intensity                |
+------------------------+-----------+-------------------------------------+
| ``p(specific)`` - |ps| | (N, F)    | probability of there being          |
|                        |           | a target-specific spot in the image |
+------------------------+-----------+-------------------------------------+

where shape provides dimensionality information about parameters, e.g., (1,) shape means
a global parameter and ``height`` with (K, N, F) shape means that *each* spot for *each*
AOI for *each* frame has an intensity parameter.

.. |ps| replace:: :math:`p(\mathsf{specific})`
.. |theta| replace:: :math:`\theta`
.. |prox| replace:: :math:`\sigma^{xy}`
.. |ld| replace:: :math:`\lambda`
.. |b| replace:: :math:`b`

Backend
-------

Tapqir's model is implemented in `Pyro`_, a Python-based probabilistic programming language
(PPL) (`Bingham et al., 2019`_). Probabilistic programming is a relatively new paradigm in
which probabilistic models are expressed in a high-level language that allows easy formulation,
modification, and automated inference.

Pyro relies on the `PyTorch`_ numeric library for vectorized math operations on GPU and automatic
differentiation. We also use `KeOps`_ library for kernel operations on the GPU without memory overflow.

Interface
---------

Tapqir is a command-line application and needs to be run in the terminal (``$`` signifies a terminal prompt).
Its command line interface is implemented in `Typer`_. The usage is ``tapqir COMMAND [OPTIONS]``. For example::

    $ tapqir fit --model cosmos --cuda

where 

* ``tapqir`` is the *program*.
* ``fit`` is the *command*.
* ``--model cosmos`` is a command *option (flag)* where ``--model`` is the option name and ``cosmos`` is the option value.
* ``--cuda`` is a Yes or No command *option (flag)* where its value is True/Yes if provided and False/No if not provided.

Some options have a one-letter version as well. For example, both ``--help`` and ``-h`` will display help.

``tapqir --help`` will display an overall help and ``tapqir COMMAND --help`` will display
a command-specific help that will show which options are available for that specific command.

*Commands* are one of the:

+------------------------+-----------------------------------+
| Command                | Short description                 |
+========================+===================================+
| | ``$ tapqir init``    | Initialize folder                 |
+------------------------+-----------------------------------+
| | ``$ tapqir glimpse`` | Extract AOIs                      |
+------------------------+-----------------------------------+
| | ``$ tapqir fit``     | Fit the data                      |
+------------------------+-----------------------------------+
| | ``$ tapqir stats``   | Calculate parameter uncertainties |
+------------------------+-----------------------------------+
| | ``$ tapqir show``    | Visualize results                 |
+------------------------+-----------------------------------+
| | ``$ tapqir log``     | Show logging info                 |
+------------------------+-----------------------------------+

Command *options* do not depend on their order. For command options that are not provided ``tapqir``
will interactively ask for the missing value::

    Tapqir model [cosmos]: cosmos

At the prompt, enter a new value by typing and then hit ENTER. To use a default value shown in ``[...]``
brackets press ENTER. For yes/no prompts type ``y`` for yes and ``n`` for no and then hit ENTER.
The default for yes/no prompt is shown in capital::

    Run computations on GPU? [Y/n]: y

Default option values are read from the ``.tapqir/config.yml`` configuration file. When the
command is run it will ask to overwrite default values (or use ``--overwrite`` flag).

To disablle all prompts use a ``--no-input`` flag (e.g., ``tapqir fit --no-input``).
This is useful after the first invocation of the command when the option values have been saved and you
want to re-run the command with the same option values.

To summarize:

1. If provided, the option value is accepted from the command line as a flag.
2. If not provided, the prompt will ask for the missing option value.
3. To disable all prompts use the ``--no-input`` flag. The program will first look for command flags and then
   for default values from the configuration file. If the required option value is missing the program will
   fail and ask to pass the information as a flag.

To escape the program use ``Ctrl-C``.

Raw input data
--------------

Tapqir analyzes a small area of interest (AOI) around each target or off-target location. AOIs (usually ``14x14`` pixels)
are extracted from raw input data. Currently Tapqir supports raw input images in `Glimpse`_ format and pre-processed
with the `imscroll`_ program:

* image data in glimpse format and header file
* aoiinfo file designating the locations of target molecules (on-target AOIs) in the binder channel
* (optional) aoiinfo file designating the off-target control locations (off-target AOIs) in the binder channel
* driftlist file recording the stage movement that took place during the experiment

We plan to extend the support to other data formats as well. Please start a `new issue`_ if you have a file format
that is not supported yet.

Workflow
--------

.. image:: ../Tapqir_workflow.png
   :alt: Tapqir workflow

.. _Ordabayev et al., 2021: https://doi.org/10.1101/2021.09.30.462536 
.. _Kinz-Thompson et al., 2021: https://doi.org/10.1146/annurev-biophys-082120-103921
.. _Bingham et al., 2019: https://jmlr.org/papers/v20/18-403.html
.. _Pyro: https://pyro.ai/
.. _PyTorch: https://pytorch.org/
.. _KeOps: https://www.kernel-operations.io/keops/index.html
.. _Typer: https://typer.tiangolo.com/
.. _Glimpse: https://github.com/gelles-brandeis/Glimpse
.. _imscroll: https://github.com/gelles-brandeis/CoSMoS_Analysis/wiki
.. _new issue: https://github.com/gelles-brandeis/tapqir/issues/new/choose
