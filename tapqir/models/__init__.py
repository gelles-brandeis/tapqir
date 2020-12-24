from tapqir.models.cosmos import Cosmos
from tapqir.models.fixedoffset import FixedOffset
from tapqir.models.model import GaussianSpot, Model

__all__ = [
    "GaussianSpot",
    "models",
    "Model",
    "FixedOffset",
    "Cosmos",
]

models = {
    FixedOffset.name: FixedOffset,
    Cosmos.name: Cosmos,
}
