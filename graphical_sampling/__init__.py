from importlib import metadata

# from .design import Design
# from .new_design import NewDesign
from .population import Population
# from .plot import plot
# from . import criteria
# from . import search
# from . import sampling
from . import clustering
from . import random
from . import index


__version__ = metadata.version("graphical_sampling")

__all__ = [
    "Population",
    "Design",
    "NewDesign",
    "criteria",
    "search",
    "sampling",
    "clustering",
    "random",
    "index",
    "plot",
]
