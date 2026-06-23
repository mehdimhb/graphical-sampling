from importlib import metadata

from .design import Design
from . import criteria
from . import search
from .population import Population


__version__ = "0.1.0"



__all__ = ["Design", "criteria", "search"]
