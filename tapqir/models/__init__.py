# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from tapqir.models.cosmos import Cosmos
from tapqir.models.hmm import HMM
from tapqir.models.model import GaussianSpot, Model

__all__ = [
    "GaussianSpot",
    "models",
    "Model",
    "Cosmos",
    "HMM",
]

models = {
    Cosmos.name: Cosmos,
    HMM.name: HMM,
}
