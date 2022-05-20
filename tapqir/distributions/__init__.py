# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import funsor
from funsor.distribution import make_dist

from tapqir.distributions.affine_beta import AffineBeta
from tapqir.distributions.ksmogn import KSMOGN

funsor.set_backend("torch")

__all__ = [
    "AffineBeta",
    "KSMOGN",
]

FunsorAffineBeta = make_dist(AffineBeta)
