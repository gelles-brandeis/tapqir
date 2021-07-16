# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from tapqir.models.active import Active, ActiveMarginal
from tapqir.models.cosmos import Cosmos, CosmosMarginal
from tapqir.models.hmm import HMM
from tapqir.models.model import Model
from tapqir.models.multispot import MultiSpot

__all__ = [
    "models",
    "Model",
    "Active",
    "ActiveMarginal",
    "Cosmos",
    "CosmosMarginal",
    "HMM",
    "MultiSpot",
]

models = {
    Active.name: Active,
    ActiveMarginal.name: ActiveMarginal,
    Cosmos.name: Cosmos,
    CosmosMarginal.name: CosmosMarginal,
    HMM.name: HMM,
    MultiSpot.name: MultiSpot,
}
