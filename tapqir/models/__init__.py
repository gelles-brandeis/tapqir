# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import Cosmos
from tapqir.models.crosstalk import Crosstalk
from tapqir.models.hmm import HMM
from tapqir.models.model import Model

__all__ = [
    "models",
    "Model",
    "Cosmos",
    "Crosstalk",
    "HMM",
]

models = {
    Cosmos.name: Cosmos,
    Crosstalk.name: Crosstalk,
    HMM.name: HMM,
}
