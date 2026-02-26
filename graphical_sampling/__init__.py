from importlib import metadata

# from .new_design import NewDesign
from .design import Design
from .population import Population
from .order import Order
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
    "Order",
    # "NewDesign",
    "criteria",
    "search",
    "sampling",
    "clustering",
    "random",
    "index",
]
