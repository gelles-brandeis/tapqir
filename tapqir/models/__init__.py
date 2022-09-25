# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import cosmos
from tapqir.models.cosmosvae import cosmosnn
from tapqir.models.crosstalk import crosstalk
from tapqir.models.hmm import hmm
from tapqir.models.model import Model

__all__ = [
    "models",
    "Model",
    "cosmosnn",
    "cosmos",
    "crosstalk",
    "hmm",
]

models = {
    cosmosnn.name: cosmosnn,
    cosmos.name: cosmos,
    crosstalk.name: crosstalk,
    hmm.name: hmm,
}
