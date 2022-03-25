Part I: Intro (Generic)
=======================

Models
------

Tapqir is a modular program that uses a chosen probabilistic model to interpret experimental data.
There currently exists only a single model, ``cosmos``, developed for analysis of simple CoSMoS
experiments. The ``cosmos`` model is for *time-independent* analysis of *single-channel* (i.e., one-binder)
data sets. Our publication (`Ordabayev et al., 2021`_) contains a comprehensive description of the
``cosmos`` model. In the future, we plan to add addional models to Tapqir, for example to integrate
hidden-Markov kinetic analysis or to handle global analysis with multiple wavelength channels.

Tapqir uses *Bayesian* models; this means that each model parameter has an associated probability
distribution (uncertainty). For those who are interested, `Kinz-Thompson et al., 2021`_ is
a nice read about Bayesian inference in the context of single-molecule data analysis.

As a consequence of Bayesian inference, Tapqir computes for each frame of each AOI the *probability*
:math:`p(\mathsf{specific})`, that a target-specific spot is present.

``cosmos`` is a physics-informed model, i.e. model parameters have a physical meaning.
For *N* AOIs per frame, *F* frames, and a maximum of *K* spots in each AOI in each frame, 
Tapqir estimates the values of the ``cosmos`` model parameters:

+-----------------+-----------+-------------------------------------+
| Parameter       | Shape     | Description                         |
+=================+===========+=====================================+
| |g| - :math:`g` | (1,)      | camera gain                         |
+-----------------+-----------+-------------------------------------+
| |sigma| - |prox|| (1,)      | proximity                           |
+-----------------+-----------+-------------------------------------+
| ``lamda`` - |ld|| (1,)      | average rate of target-nonspecific  |
|                 |           | binding                             |
+-----------------+-----------+-------------------------------------+
| ``pi`` - |pi|   | (1,)      | average binding probability of      |
|                 |           | target-specific binding             |
+-----------------+-----------+-------------------------------------+
| |bg| - |b|      | (N, F)    | background intensity                |
+-----------------+-----------+-------------------------------------+
| |z| - :math:`z` | (N, F)    | target-specific spot presence       |
+-----------------+-----------+-------------------------------------+
| |t| - |theta|   | (N, F)    | target-specific spot index          |
+-----------------+-----------+-------------------------------------+
| |m| - :math:`m` | (K, N, F) | spot presence indicator             |
+-----------------+-----------+-------------------------------------+
| |h| - :math:`h` | (K, N, F) | spot intensity                      |
+-----------------+-----------+-------------------------------------+
| |w| - :math:`w` | (K, N, F) | spot width                          |
+-----------------+-----------+-------------------------------------+
| |x| - :math:`x` | (K, N, F) | spot position on x-axis             |
+-----------------+-----------+-------------------------------------+
| |y| - :math:`y` | (K, N, F) | spot position on y-axis             |
+-----------------+-----------+-------------------------------------+
| |D| - :math:`D` | |shape|   | observed images                     |
+-----------------+-----------+-------------------------------------+

.. |ps| replace:: :math:`p(\mathsf{specific})`
.. |theta| replace:: :math:`\theta`
.. |prox| replace:: :math:`\sigma^{xy}`
.. |ld| replace:: :math:`\lambda`
.. |b| replace:: :math:`b`
.. |shape| replace:: (N, F, P, P)
.. |sigma| replace:: ``proximity``
.. |bg| replace:: ``background``
.. |h| replace:: ``height``
.. |w| replace:: ``width``
.. |D| replace:: ``data``
.. |m| replace:: ``m``
.. |z| replace:: ``z``
.. |t| replace:: ``theta``
.. |x| replace:: ``x``
.. |y| replace:: ``y``
.. |pi| replace:: :math:`\pi`
.. |g| replace:: ``gain``

where "shape" is the dimensionality of the parameters, e.g., (1,) shape means a scalar
parameter and (K, N, F) shape means that *each* spot in *each* AOI in *each* frame
has a separate value of the parameter. `Ordabayev et al., 2021`_ has a more detailed
description of the parameters.

Some basic Linux commands
^^^^^^^^^^^^^^^^^^^^^^^^^

For a quick reference, some commonly used Linux commands:

1. ``pwd`` - Print the name of the current working directory.
2. ``ls`` - List files and folders.
3. ``cd`` - Change the working directory (e.g., ``cd Downloads``)
4. ``mkdir`` - Create a folder (e.g., ``mkdir new_folder``). Tip: try to avoid spaces in file & folder
   names because spaces need a special escape character ``\``.
5. ``rm`` - Delete files. Use ``rm -r`` to delete folders. Be careful, files delted with ``rm`` command
   do not go to the recycle bin and are permanently deleted!
6. ``cp`` - Copy files. Usage is ``cp <from> <to>``.
7. ``mv`` - Move or rename files. Usage is ``mv <from> <to>``.
8. Use double ``[TAB]`` for command or filename completion.

Input data
----------

Tapqir analyzes a small area of interest (AOI) around each target or off-target location. AOIs (usually ``14x14`` pixels)
are extracted from raw input data. Currently Tapqir supports raw input images in `Glimpse`_ format and pre-processing
information files from the `imscroll`_ program:

* folder containing image data in glimpse format and header files
* driftlist file recording the stage movement that took place during the experiment
* aoiinfo file designating target molecule locations in the binder channel
* (optional) aoiinfo file designating off-target locations in the binder channel

We plan to extend the support to other data formats. Please start a `new issue`_ if you would like to work with us 
to extend support to file formats used in your processing pipeline.

Workflow
--------

The following diagram shows the steps in a Tapqir data processing run (using the ``cosmos`` model), the Tapqir command
used to run each step, and the input files used and output files produced (color highlights) in each step. All the
Tapqir commands for a single processing run should be run in the same default working directory (``new_folder`` in
the diagram) in order to keep the files associated with the run organized in a single location.

.. image:: ../Tapqir_workflow.png
   :alt: Tapqir workflow

.. _Ordabayev et al., 2021: https://doi.org/10.7554/eLife.73860
.. _Kinz-Thompson et al., 2021: https://doi.org/10.1146/annurev-biophys-082120-103921
.. _Bingham et al., 2019: https://jmlr.org/papers/v20/18-403.html
.. _Typer: https://typer.tiangolo.com/
.. _YAML: https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html
.. _Glimpse: https://github.com/gelles-brandeis/Glimpse
.. _imscroll: https://github.com/gelles-brandeis/CoSMoS_Analysis/wiki
.. _new issue: https://github.com/gelles-brandeis/tapqir/issues/new/choose
