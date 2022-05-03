# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from tapqir.models.cosmos import Cosmos
from tapqir.models.cosmos2 import Cosmos2
from tapqir.models.ebcosmos import EBCosmos
from tapqir.models.hmm import HMM
from tapqir.models.model import Model
from tapqir.models.mscosmos import MSCosmos
from tapqir.models.mshmm import MSHMM

__all__ = [
    "models",
    "Model",
    "Cosmos",
    "Cosmos2",
    "MSCosmos",
    "EBCosmos",
    "HMM",
    "MSHMM",
]

models = {
    Cosmos.name: Cosmos,
    Cosmos2.name: Cosmos2,
    MSCosmos.name: MSCosmos,
    EBCosmos.name: EBCosmos,
    HMM.name: HMM,
    MSHMM.name: MSHMM,
}
