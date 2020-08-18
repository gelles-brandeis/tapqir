from cosmos.models.model import Model
from cosmos.models.tracker import Tracker
from cosmos.models.spot import Spot

__all__ = [
    "models",
    "Model",
    "Spot",
    "Tracker"
]

models = {
    "spot": Spot,
    "tracker": Tracker
}
