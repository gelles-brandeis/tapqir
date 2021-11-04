# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import Cosmos
from tapqir.models.hmm import HMM
from tapqir.models.model import Model

__all__ = [
    "models",
    "Model",
    "Cosmos",
    "HMM",
]

models = {
    Cosmos.name: Cosmos,
    HMM.name: HMM,
}
