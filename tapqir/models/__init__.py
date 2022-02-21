# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import Cosmos
from tapqir.models.crosstalk import Crosstalk
from tapqir.models.crosstalkhmm2 import CrosstalkHMM
from tapqir.models.hmm import HMM
from tapqir.models.mshmm import MSHMM
from tapqir.models.model import Model

__all__ = [
    "models",
    "Model",
    "Cosmos",
    "Crosstalk",
    "CrosstalkHMM",
    "HMM",
    "MSHMM",
]

models = {
    Cosmos.name: Cosmos,
    Crosstalk.name: Crosstalk,
    CrosstalkHMM.name: CrosstalkHMM,
    HMM.name: HMM,
    MSHMM.name: MSHMM,
}
