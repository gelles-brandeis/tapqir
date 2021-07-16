# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import funsor
from funsor.distribution import make_dist

from tapqir.distributions.affine_beta import AffineBeta
from tapqir.distributions.convoluted_gamma import ConvolutedGamma
from tapqir.distributions.kspotgammanoise import KSpotGammaNoise

funsor.set_backend("torch")

__all__ = [
    "AffineBeta",
    "ConvolutedGamma",
    "KSpotGammaNoise",
]

FunsorAffineBeta = make_dist(AffineBeta)
FunsorConvolutedGamma = make_dist(ConvolutedGamma)
