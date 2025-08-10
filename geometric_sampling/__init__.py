from importlib import metadata

from .design import Design
from . import criteria
from . import search
from . import sampling
from . import clustering
from . import random
from . import measure

try:
    __version__ = metadata.version("geometric_sampling")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0-dev"  # Development version

__all__ = [
    "Design",
    "criteria",
    "search",
    "sampling",
    "clustering",
    "random",
    "measure",
]
