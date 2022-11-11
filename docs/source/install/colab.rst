Install on Google Colab
=======================

Colab notebooks start in a fresh environment and thus require installation for each new
start.

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

.. _a new notebook: https://colab.research.google.com/?utm_source=scs-index 
