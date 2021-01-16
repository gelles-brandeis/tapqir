import funsor
from funsor.distribution import make_dist

from tapqir.distributions.affine_beta import AffineBeta
from tapqir.distributions.convoluted_gamma import ConvolutedGamma
from tapqir.distributions.convoluted_normal import ConvolutedNormal
from tapqir.distributions.fixed_offset_gamma import FixedOffsetGamma
from tapqir.distributions.multi_modal import MultiModal

funsor.set_backend("torch")

__all__ = [
    "AffineBeta",
    "ConvolutedGamma",
    "ConvolutedNormal",
    "FixedOffsetGamma",
    "MultiModal",
]

FunsorAffineBeta = make_dist(AffineBeta)
FunsorConvolutedGamma = make_dist(ConvolutedGamma)
FunsorConvolutedNormal = make_dist(ConvolutedNormal)
