# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from tapqir.models.cosmos import Cosmos
from tapqir.models.hmm import HMM
from tapqir.models.model import Model
from tapqir.models.multispot import MultiSpot

__all__ = [
    "models",
    "Model",
    "Cosmos",
    "HMM",
    "MultiSpot",
]

models = {
    Cosmos.name: Cosmos,
    HMM.name: HMM,
    MultiSpot.name: MultiSpot,
}
