# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from tapqir.models.cosmos import Cosmos, CosmosMarginal
from tapqir.models.cosmosvae import CosmosVAE, CosmosVAEMarginal
from tapqir.models.hmm import HMM
from tapqir.models.model import Model
from tapqir.models.multispot import MultiSpot

__all__ = [
    "models",
    "Model",
    "Cosmos",
    "CosmosMarginal",
    "CosmosVAE",
    "CosmosMarginalVAE",
    "HMM",
    "MultiSpot",
]

models = {
    Cosmos.name: Cosmos,
    CosmosMarginal.name: CosmosMarginal,
    CosmosVAE.name: CosmosVAE,
    CosmosVAEMarginal.name: CosmosVAEMarginal,
    HMM.name: HMM,
    MultiSpot.name: MultiSpot,
}
