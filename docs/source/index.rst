Documentation
=============

:mod:`tapqir` is software to analyze images from single-molecule fluorescence colocalization experiments 
using a Bayesian statistics-based image classification method.

:mod:`tapqir` is implemented in `Pyro <https://pyro.ai/>`_, a Python-based probabilistic programming language.

:mod:`tapqir` is open source software licensed under the `GNU General Public License, version 3 <http://www.gnu.org/licenses/gpl-3.0.txt>`_.

**Author**:
Yerdos Ordabayev;
Department of Biochemistry, MS009;
Brandeis University;
PO Box 549110;
Waltham, MA 02454-9110;
ordabayev@brandeis.edu

**Acknowledgements**:  Development was funded by grants from `NIGMS <http://www.nigms.nih.gov>`_.

**Citation**:  If you publish research that uses this software, you can cite https://github.com/gelles-brandeis/tapqir

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: _static/fi-rr-cloud-download.svg

    Installation
    ^^^^^^^^^^^^

    New to *tapqir*? Check out the installation guide.

    +++

    .. link-button:: install
            :type: ref
            :text: To the installation guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/fi-rr-book-alt.svg

    Tutorial
    ^^^^^^^^

    The tutorial provides in-depth information on running tapqir models.

    +++

    .. link-button:: notebooks/tutorial
            :type: ref
            :text: To the tutorial
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/fi-rr-layers.svg

    API reference
    ^^^^^^^^^^^^^

    The reference guide contains a detailed description of
    the tapqir API.

    +++

    .. link-button:: api
            :type: ref
            :text: To the API reference
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/fi-rr-gallery.svg

    Examples
    ^^^^^^^^

    Examples of analysis of experimental data using tapqir.

    +++

    .. link-button:: examples/index
            :type: ref
            :text: To the examples
            :classes: btn-block btn-secondary stretched-link

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :hidden:

   install
   notebooks/tutorial
   api
   examples/index
   user_guide/commands
