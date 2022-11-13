Install on Google Colab
=======================

`Google Colab`_ is a cloud computing environment provided by Google as an inexpensive way to access powerful,
GPU-equipped computers to do machine learning computations like Tapqir. If you are new to Tapqir, it will be
easiest to run it in Colab.

To get started, get a Colab account (if you or your lab don’t already have one). Grab a credit card and head to
https://colab.research.google.com/signup. We use a Colab Pro + account ($50/month), but even
a Colab Pro account ($10 / month) may work.

Colab notebooks start in a fresh environment and thus require Tapqir installation for each new
start. New users should use `example notebook`_ in our :doc:`tutorial </tutorials/part_ii_colab>` - which already contains the commands to install Tapqir.

Installing Tapqir in a new notebook (Advanced)
----------------------------------------------

Advanced Colab users can create a new notebook

.. note:: Colab notebooks are Python environments but also allow running shell
   commands by prepending ``!`` to them. Installation commands and all tapqir commands
   are shell commands and therefore need a prepended ``!`` sign.

1. Start `a new notebook`_.

2. Before installing ``tapqir``, switch runtime to GPU (in the menu select ``Runtime ->
   Change runtime type -> GPU``)

3. Mount Google Drive to be able to save analysis output.

4. Install ``tapqir`` using pip (to avoid clutter in the notebook installation
   output is silenced)::

    !pip install --quiet tapqir > install.log

5. After installing ``tapqir``, restart the runtime (in the menu click ``Runtime -> Restart runtime``)

Check version
-------------

To check Tapqir version run::

   !tapqir --version

.. _Google Colab: https://research.google.com/colaboratory/faq.html
.. _a new notebook: https://colab.research.google.com/?utm_source=scs-index 
