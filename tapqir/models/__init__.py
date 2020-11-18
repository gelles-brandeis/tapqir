from tapqir.models.model import GaussianSpot
from tapqir.models.model import Model
from tapqir.models.cosmos import Cosmos
from tapqir.models.fixedoffset import FixedOffset

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
