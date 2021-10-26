Disclaimer
==========

This is an alpha version of the program. It may contain bugs and is subject to change.

Tapqir
======

Bayesian analysis of co-localization single-molecule microscopy image data.

.. |ci| image:: https://github.com/gelles-brandeis/tapqir/workflows/build/badge.svg
  :target: https://github.com/gelles-brandeis/tapqir/actions

.. |docs| image:: https://readthedocs.org/projects/tapqir/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://tapqir.readthedocs.io/en/latest/?badge=latest

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/ambv/black
  
.. |DOI| image:: https://img.shields.io/badge/DOI-10.1101%2F2021.09.30.462536-blue
   :target: https://doi.org/10.1101/2021.09.30.462536
   :alt: DOI

|DOI| |ci| |docs| |black|

`Preprint <https://doi.org/10.1101/2021.09.30.462536>`_ |
`Documentation <https://tapqir.readthedocs.io/en/latest/>`_ |
`Discussions <https://github.com/gelles-brandeis/tapqir/discussions/>`_

**Tapqir** is an **open-source** (Python) program for modeling and analysis of single-molecule image data.
Key features:

1. Tapqir's analysis method is based on a **holistic**, **physics-informed** causal model of CoSMoS image data.
2. Instead of yielding a binary "spot/no spot" classification, Tapqir calculates the **probability** of a target-specific spot being present.
3. Tapqir's model is implemented in the Python-based probabilistic programming language `Pyro <https://pyro.ai/>`_.
4. Tapqir has a simple command-line interface implemented in `Typer <https://typer.tiangolo.com/>`_.

Installation
============

OS-specific installation instructions are `here <https://tapqir.readthedocs.io/en/latest/install/index.html>`_.

To install using **pip**, run::

  pip install git+https://github.com/gelles-brandeis/tapqir.git


Documentation
=============

Documentation and tutorial available at `tapqir.readthedocs.io <https://tapqir.readthedocs.io/>`_. 
Please note that the documentation is not yet complete and may not be up to date.
  
Tapqir workflow
===============

.. image:: docs/source/Tapqir_workflow.png

License
=======

This project is licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

By submitting a pull request to this project, you agree to license your contribution under the Apache license version 2.0 to this project.

Citation
========

If you use Tapqir, please consider citing our preprint:

|DOI|

Ordabayev YA, Friedman LJ, Gelles J, Theobald DL. *Bayesian machine learning analysis of single-molecule fluorescence colocalization images*.
bioRxiv. 2021 Oct.
