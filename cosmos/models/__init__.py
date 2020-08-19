from cosmos.models.model import Model
from cosmos.models.tracker import Tracker
from cosmos.models.spot import Spot
from cosmos.models.test import Test

__all__ = [
    "models",
    "Model",
    "Spot",
    "Tracker",
    "Test"
]

models = {
    "spot": Spot,
    "tracker": Tracker,
    "test": Test
}
