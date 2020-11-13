from tapqir.models.model import GaussianSpot
from tapqir.models.model import Model
from tapqir.models.spotdetection import SpotDetection
from tapqir.models.fixedoffset import FixedOffset
from tapqir.models.cosmos import Cosmos

__all__ = [
    "GaussianSpot",
    "models",
    "Model",
    "SpotDetection",
    "FixedOffset",
    "Cosmos",
]

models = {
    SpotDetection.name: SpotDetection,
    FixedOffset.name: FixedOffset,
    Cosmos.name: Cosmos,
}
