# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import cosmos
from tapqir.models.cosmosvae import cosmosvae
from tapqir.models.crosstalk import crosstalk
from tapqir.models.hmm import hmm
from tapqir.models.model import Model

__all__ = [
    "models",
    "Model",
    "cosmos",
    "cosmosvae",
    "crosstalk",
    "hmm",
]

models = {
    cosmosvae.name: cosmosvae,
    cosmos.name: cosmos,
    crosstalk.name: crosstalk,
    hmm.name: hmm,
}
