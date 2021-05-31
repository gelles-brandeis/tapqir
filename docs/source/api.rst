API
===

.. currentmodule:: tapqir

Models
~~~~~~

.. currentmodule:: tapqir

.. autosummary::
   :toctree: reference/
   :nosignatures:

   models.GaussianSpot
   models.Model
   models.Cosmos
   models.HMM


Data formats & Data preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: tapqir

.. autosummary::
   :toctree: reference/
   :nosignatures:

   utils.dataset.CosmosData
   utils.dataset.OffsetData
   utils.dataset.CosmosDataset
   imscroll.GlimpseDataset
   imscroll.read_glimpse


Command line tools
~~~~~~~~~~~~~~~~~~

.. currentmodule:: tapqir

.. autosummary::
   :toctree: reference/
   :nosignatures:

   commands.Config
   commands.Glimpse
   commands.Fit
   commands.Show
   commands.Save


Distributions
~~~~~~~~~~~~~

For details on the Pyro distribution interface, see :class:`pyro.distributions.TorchDistribution`.

.. currentmodule:: tapqir

.. autosummary::
   :toctree: reference/
   :nosignatures:

   distributions.AffineBeta
   distributions.ConvolutedGamma
