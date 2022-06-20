# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import cosmos
from tapqir.models.cosmosvae import cosmosnn
from tapqir.models.crosstalk import crosstalk
from tapqir.models.hmm import HMM
from tapqir.models.model import Model

__all__ = [
    "models",
    "Model",
    "cosmosnn",
    "cosmos",
    "crosstalk",
    "HMM",
]

models = {
    cosmosnn.__name__: cosmosnn,
    cosmos.__name__: cosmos,
    crosstalk.__name__: crosstalk,
    HMM.name: HMM,
}
