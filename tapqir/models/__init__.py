from tapqir.models.model import GaussianSpot
from tapqir.models.model import Model
from tapqir.models.spotdetection import SpotDetection
from tapqir.models.fixedoffset import FixedOffset
# from tapqir.models.empirical import Empirical

__all__ = [
    "GaussianSpot",
    "models",
    "Model",
    "SpotDetection",
    "FixedOffset",
    # "Empirical",
]

models = {
    SpotDetection.name: SpotDetection,
    FixedOffset.name: FixedOffset,
    # Empirical.name: Empirical,
}
