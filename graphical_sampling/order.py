import copy
from dataclasses import dataclass, field

import numpy as np

from .population import Population


@dataclass
class Zone:
    label: int
    indices: list[int] = field(default_factory=list)


@dataclass
class Floor:
    index: int | None = None
    percentage: float | None = None


@dataclass
class Ceil:
    index: int | None = None
    percentage: float | None = None


@dataclass
class Cluster:
    label: int
    zones: list[Zone] = field(default_factory=list)
    floor: Floor | None = None
    ceil: Ceil | None = None


class Order:
    def __init__(self, population: Population):
        """
        Base constructor. Use the class methods `from_indices` or `from_clusters`
        to properly instantiate and build the order.
        """
        self.pop = population
        self._order: np.ndarray = np.array([], dtype=int)
        self._fixed_ids: set[int] = set()
        self.clusters: list[Cluster] | None = None

    @classmethod
    def from_indices(cls, population: Population, random: bool = False) -> Order:
        instance = cls(population)
        if random:
            rng = np.random.default_rng()
            instance._order = rng.permutation(population.indices)
        else:
            instance._order = np.copy(population.indices)
        return instance

    @classmethod
    def from_clusters(cls, population: Population, clusters: list[Cluster]) -> Order:
        instance = cls(population)
        instance.clusters = clusters
        instance._rebuild_order()
        return instance

    def _rebuild_order(self):
        """Helper method to construct the _order array and fixed IDs based on current clusters."""
        if not self.clusters:
            return

        order_list: list[int] = []
        fixed_ids: set[int] = set()
        seen = {-1}

        def add_id(_id: int):
            if _id not in seen:
                seen.add(_id)
                order_list.append(_id)

        for cluster in self.clusters:
            if cluster.floor is not None and cluster.floor.index is not None:
                add_id(cluster.floor.index)
                fixed_ids.add(cluster.floor.index)

            for zone in cluster.zones:
                for item_id in zone.indices:
                    add_id(item_id)

            if cluster.ceil is not None and cluster.ceil.index is not None:
                add_id(cluster.ceil.index)
                fixed_ids.add(cluster.ceil.index)

        self._order = np.array(order_list, dtype=int)
        self._fixed_ids = fixed_ids

    def change(self, num_clusters: int, num_zones: int, num_changes: int, num_zone_changes: int):
        """
        Randomly changes the position of indices inside the zones, AND/OR
        changes the order of the zones themselves within the clusters.
        """
        if not self.clusters:
            raise ValueError("Order must be initialized with from_clusters to use the change method.")

        rng = np.random.default_rng()

        # --- Phase 1: Item-level changes inside zones ---
        if num_changes > 0:
            # Filter out clusters that have no zones to avoid indexing errors
            valid_clusters_items = [c for c in self.clusters if len(c.zones) > 0]
            if valid_clusters_items:
                # Select random clusters safely
                c_idxs_items = rng.choice(len(valid_clusters_items),
                                          size=min(num_clusters, len(valid_clusters_items)),
                                          replace=False)
                selected_clusters_items = [valid_clusters_items[i] for i in c_idxs_items]

                for cluster in selected_clusters_items:
                    # We need at least 2 items in a zone to actually "change position"
                    valid_zones = [z for z in cluster.zones if len(z.indices) > 1]
                    if not valid_zones:
                        continue

                    z_idxs = rng.choice(len(valid_zones), size=min(num_zones, len(valid_zones)), replace=False)
                    selected_zones = [valid_zones[i] for i in z_idxs]

                    for zone in selected_zones:
                        for _ in range(num_changes):
                            old_idx = int(rng.integers(0, len(zone.indices)))
                            item = zone.indices.pop(old_idx)
                            new_idx = int(rng.integers(0, len(zone.indices) + 1))
                            zone.indices.insert(new_idx, item)

        # --- Phase 2: Zone-level changes inside clusters ---
        if num_zone_changes > 0:
            # We need at least 2 zones in a cluster to change their order
            valid_clusters_zones = [c for c in self.clusters if len(c.zones) > 1]
            if valid_clusters_zones:
                # Select a potentially different set of random clusters for zone moving
                c_idxs_zones = rng.choice(len(valid_clusters_zones),
                                          size=min(num_clusters, len(valid_clusters_zones)),
                                          replace=False)
                selected_clusters_zones = [valid_clusters_zones[i] for i in c_idxs_zones]

                for cluster in selected_clusters_zones:
                    for _ in range(num_zone_changes):
                        # Pick a zone, remove it, and insert it at a new random index
                        old_idx = int(rng.integers(0, len(cluster.zones)))
                        moved_zone = cluster.zones.pop(old_idx)

                        new_idx = int(rng.integers(0, len(cluster.zones) + 1))
                        cluster.zones.insert(new_idx, moved_zone)

        # Rebuild the _order array to reflect all new states
        self._rebuild_order()

    def get(self) -> np.ndarray:
        return self._order

    @property
    def fixed_ids(self) -> set[int]:
        return self._fixed_ids

    def copy(self) -> Order:
        """
        Creates an independent duplicate of the Order.
        Mutating the copy will not affect the original Order's clusters, zones, or arrays.
        """
        # Create a fresh instance. We pass the same population reference
        # since the population itself isn't what we are mutating.
        new_instance = Order(self.pop)

        # Copy the numpy array and the set of fixed IDs
        new_instance._order = np.copy(self._order)
        new_instance._fixed_ids = self._fixed_ids.copy()

        # Safely deep-copy the clusters so all nested zones/lists are completely separate
        if self.clusters is not None:
            new_instance.clusters = copy.deepcopy(self.clusters)
        else:
            new_instance.clusters = None

        return new_instance
