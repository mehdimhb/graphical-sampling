import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from .population import Population


@dataclass
class Zone:
    label: int
    ids: List[int] = field(default_factory=list)


@dataclass
class Floor:
    index: Optional[int] = None
    percentage: Optional[float] = None


@dataclass
class Ceil:
    index: Optional[int] = None
    percentage: Optional[float] = None


@dataclass
class Cluster:
    label: int
    zones: List[Zone] = field(default_factory=list)
    floor: Optional[Floor] = None
    ceil: Optional[Ceil] = None


class Order:
    def __init__(self, population: Population):
        """
        Base constructor. Use the class methods `from_indices` or `from_clusters`
        to properly instantiate and build the order.
        """
        self.pop = population
        self._order: np.ndarray = np.array([], dtype=int)
        self._fixed_ids: set[int] = set()

    @classmethod
    def from_indices(cls, population: Population, random: bool = False) -> "Order":
        instance = cls(population)
        if random:
            rng = np.random.default_rng()
            instance._order = rng.permutation(population.indices)
        else:
            instance._order = np.copy(population.indices)
        return instance

    @classmethod
    def from_clusters(cls, population: Population, clusters: List[Cluster]) -> "Order":
        instance = cls(population)
        order_list: List[int] = []
        fixed_ids: set[int] = set()
        seen = {-1}

        def add_id(_id: int):
            if _id not in seen:
                seen.add(_id)
                order_list.append(_id)

        for cluster in clusters:
            if cluster.floor is not None and cluster.floor.index is not None:
                add_id(cluster.floor.index)
                fixed_ids.add(cluster.floor.index)

            for zone in cluster.zones:
                for item_id in zone.ids:
                    add_id(item_id)

            if cluster.ceil is not None and cluster.ceil.index is not None:
                add_id(cluster.ceil.index)
                fixed_ids.add(cluster.ceil.index)

        instance._order = np.array(order_list, dtype=int)
        instance._fixed_ids = fixed_ids
        return instance

    def get(self) -> np.ndarray:
        return self._order

    @property
    def fixed_ids(self) -> set[int]:
        return self._fixed_ids
