# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import Cosmos
from tapqir.models.cthmm import CTHMM
from tapqir.models.hmm import HMM
from tapqir.models.model import Model
from tapqir.models.multicolor import MultiColor
from tapqir.models.multispot import MultiSpot

__all__ = [
    "models",
    "Model",
    "Cosmos",
    "HMM",
    "CTHMM",
    "MultiColor",
    "MultiSpot",
]

models = {
    Cosmos.name: Cosmos,
    HMM.name: HMM,
    CTHMM.name: CTHMM,
    MultiColor.name: MultiColor,
    MultiSpot.name: MultiSpot,
}
