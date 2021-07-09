# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from tapqir.models.cosmos import Cosmos, CosmosMarginal
from tapqir.models.hmm import HMM
from tapqir.models.model import GaussianSpot, Model
from tapqir.models.multispot import MultiSpot

__all__ = [
    "GaussianSpot",
    "models",
    "Model",
    "Cosmos",
    "CosmosMarginal",
    "HMM",
    "MultiSpot",
]

models = {
    Cosmos.name: Cosmos,
    CosmosMarginal.name: CosmosMarginal,
    HMM.name: HMM,
    MultiSpot.name: MultiSpot,
}
