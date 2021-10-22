Tapqir Documentation
====================

Tapqir is software to analyze images from single-molecule fluorescence colocalization experiments 
using a Bayesian statistics-based image classification method. Tapqir is implemented in
`Pyro <https://pyro.ai/>`_, a Python-based probabilistic programming language.

**License**: Tapqir is open source software licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

**Author**:
Yerdos Ordabayev;
Department of Biochemistry, MS009;
Brandeis University;
PO Box 549110;
Waltham, MA 02454-9110;
ordabayev@brandeis.edu

**Acknowledgements**:  Development was funded by grants from `NIGMS <http://www.nigms.nih.gov>`_.

**Citation**:  If you publish research that uses this software, you can cite our preprint::

  @article{ordabayev2021bayesian,
    title={Bayesian machine learning analysis of single-molecule fluorescence colocalization images},
    author={Ordabayev, Yerdos A and Friedman, Larry J and Gelles, Jeff and Theobald, Douglas},
    journal={bioRxiv},
    year={2021},
    publisher={Cold Spring Harbor Laboratory}
  }

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: _static/fi-rr-cloud-download.svg

    Installation
    ^^^^^^^^^^^^

    New to *tapqir*? Check out the installation guide.

    +++

    .. link-button:: install/index
            :type: ref
            :text: To the installation guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/fi-rr-book-alt.svg

    User Guide
    ^^^^^^^^^^

    The user guide provides in-depth information on running tapqir models.

    +++

    .. link-button:: user_guide/index
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/fi-rr-layers.svg

    Command Reference
    ^^^^^^^^^^^^^^^^^

    The reference guide contains a detailed description of
    the tapqir commands.

    +++

    .. link-button:: commands/index
            :type: ref
            :text: To the command reference
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/fi-rr-gallery.svg

    Tutorials
    ^^^^^^^^^

    Examples of analysis of experimental data using tapqir.

    +++

    .. link-button:: tutorials/index
            :type: ref
            :text: To the examples
            :classes: btn-block btn-secondary stretched-link

.. toctree::
   :titlesonly:
   :hidden:

   install/index
   user_guide/index
   tutorials/index
   commands/index
