General Information
===================

About Tapqir
------------

Single-molecule fluorescence microscopy is widely used in vitro to study the biochemical and physical mechanisms
of the protein and nucleic acid macromolecular “machines” that perform essential biological functions.  The simplest
such technique is multi-wavelength colocalization, which is sometimes called CoSMoS (co-localization single-molecule
spectroscopy).  In CoSMoS, formation and/or dissociation of molecular complexes is observed by total internal
reflection fluorescence (TIRF) or other forms of single-molecule fluorescence microscopy by observing the colocalization
of two or more macromolecular components each labeled with a different color of fluorescent dye.  Analysis of the dynamics
observed in the microscope is then used to define the quantitative kinetic mechanism of the process being studied.

Reliable analysis of CoSMoS data remains a significant challenge to the effective and more widespread use of the
technique. Existing analysis methods are at least partially subjective and require painstaking manual tuning.
Data analysis is usually the slowest and most laborious part of a CoSMoS project.  

Tapqir is a computer program for rigorous statistical classification and analysis of image data from CoSMoS experiments.
The program has multiple advantageous features:

* Tapqir maximizes extraction of useful information by globally fitting experimental images to a causal probabilistic
  model that explicitly accounts for all important physical and chemical aspects of CoSMoS image formation. The fitting
  employs Bayesian inference, incorporating appropriate levels of prior knowledge (or lack of knowledge) for all parameters.

* Existing methods produce a binary spot/no-spot classification that does not convey the uncertainties inherent in
  interpreting the low signal-to-noise single-molecule images.  Tapqir instead produces spot probability estimates that
  accurately convey experimental uncertainty at each individual time point.  These probability estimates can then be
  used to perform more reliable downstream kinetic and thermodynamic analyses. 

* Tapqir has been thoroughly validated by measuring its performance on simulated image datasets.

* Tapqir is a fully objective method; we have shown that it works without manual parameter tweaking on both simulated and
  experiment-derived data sets with a wide range of signal, noise, and non-specific binding characteristics. 

Citation
--------

Initial development and validation of Tapqir is described in

|DOI|

Ordabayev YA, Friedman LJ, Gelles J, Theobald DL. *Bayesian machine learning analysis of single-molecule
fluorescence colocalization images*. bioRxiv. 2021 Oct.

If you publish work that uses Tapqir, please consider citing this article. 

License
-------

This project is licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

By submitting a pull request to this project, you agree to license your contribution under the Apache
License 2.0 to this project.

Open soucre community
---------------------

We are committed to working with users from other labs who want to incorporate Tapqir into their image processing
pipelines. We welcome help from outside users to:

* Test Tapqir with other microscope or camera (e.g., sCMOS) technologies 
* Make Tapqir work with other data file formats and preprocessing pipelines
* Export Tapqir results to different post-processing software

Backend
-------

Tapqir's model is implemented in `Pyro`_, a Python-based probabilistic programming language
(PPL) (`Bingham et al., 2019`_). Probabilistic programming is a relatively new paradigm in
which probabilistic models are expressed in a high-level language that allows easy formulation,
modification, and automated inference.

Pyro relies on the `PyTorch`_ numeric library for vectorized math operations on GPU and
automatic differentiation. We also use `KeOps`_ library for kernel operations on the GPU
without memory overflow.

.. _Bingham et al., 2019: https://jmlr.org/papers/v20/18-403.html
.. _Pyro: https://pyro.ai/
.. _KeOps: https://www.kernel-operations.io/keops/index.html
.. _PyTorch: https://pytorch.org/
.. |DOI| image:: https://img.shields.io/badge/DOI-10.1101%2F2021.09.30.462536-blue
   :target: https://doi.org/10.1101/2021.09.30.462536
   :alt: DOI
