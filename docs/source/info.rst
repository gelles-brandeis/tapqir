General Information
===================

Backend
-------

Tapqir's model is implemented in `Pyro`_, a Python-based probabilistic programming language
(PPL) (`Bingham et al., 2019`_). Probabilistic programming is a relatively new paradigm in
which probabilistic models are expressed in a high-level language that allows easy formulation,
modification, and automated inference.

Pyro relies on the `PyTorch`_ numeric library for vectorized math operations on GPU and
automatic differentiation. We also use `KeOps`_ library for kernel operations on the GPU
without memory overflow.

License
-------

This project is licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

By submitting a pull request to this project, you agree to license your contribution under the Apache
license version 2.0 to this project.

Citation
--------

If you use Tapqir, please consider citing our preprint:

|DOI|

Ordabayev YA, Friedman LJ, Gelles J, Theobald DL. *Bayesian machine learning analysis of single-molecule
fluorescence colocalization images*. bioRxiv. 2021 Oct.

.. _Bingham et al., 2019: https://jmlr.org/papers/v20/18-403.html
.. _Pyro: https://pyro.ai/
.. _KeOps: https://www.kernel-operations.io/keops/index.html
.. _PyTorch: https://pytorch.org/
.. |DOI| image:: https://img.shields.io/badge/DOI-10.1101%2F2021.09.30.462536-blue
   :target: https://doi.org/10.1101/2021.09.30.462536
   :alt: DOI
