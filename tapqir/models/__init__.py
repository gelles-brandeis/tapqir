# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import cosmos
from tapqir.models.hmm import HMM
from tapqir.models.model import Model

__all__ = [
    "models",
    "Model",
    "cosmos",
    "HMM",
]

models = {
    cosmos.__name__: cosmos,
    HMM.__name__: HMM,
}
