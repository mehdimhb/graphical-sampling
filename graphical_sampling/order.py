from __future__ import annotations
import copy
from dataclasses import dataclass, field

import numpy as np

from .population import Population


@dataclass
class Zone:
    _shares: np.ndarray = field(default_factory=np.ndarray)
    _indices: np.ndarray = field(default_factory=np.ndarray)
    sort: list[int] = field(default_factory=list)

    @property
    def shares(self) -> np.ndarray:
        return self._shares[self.sort]

    @property
    def indices(self) -> np.ndarray:
        return self._indices[self.sort]


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
        self._order: np.ndarray = np.array([], dtype=float)
        self._fixed_ids: set[int] = set()
        self.clusters: list[Cluster] | None = None
        self.num_zones: int = 0
        self.num_splits: int = 1

    @classmethod
    def from_indices(cls, population: Population, permutation: bool = False, num_splits: int = 1) -> Order:
        instance = cls(population)

        if permutation:
            rng = np.random.default_rng()
            indices = np.repeat(rng.permutation(population.indices), num_splits)
        else:
            indices = np.repeat(np.copy(population.indices), num_splits)

        shares = np.repeat(np.ones_like(population.indices), num_splits) / num_splits

        instance._order = np.column_stack([indices, shares])
        instance.num_zones = 1
        instance.num_splits = num_splits
        return instance

    @classmethod
    def from_clusters(
            cls,
            population: Population,
            clusters: list[Cluster],
            zone_strategy: str | None = None,
            point_strategy: str | None = None,
            num_splits: int = 1
    ) -> Order:
        num_zones = len(clusters[0].zones)
        for cluster in clusters:
            assert len(cluster.zones) == num_zones, 'Number of zones must be the same across all clusters.'

        instance = cls(population)
        if zone_strategy is not None or point_strategy is not None:
            instance.clusters = instance._modify_clusters(population, clusters, zone_strategy, point_strategy)
        else:
            instance.clusters = clusters
        instance._build_order(num_splits)
        instance.num_zones = num_zones
        instance.num_splits = num_splits
        return instance

    def _modify_clusters(
            self, pop: Population, clusters: list[Cluster], zone_strategy: str | None, point_strategy: str | None
    ) -> list[Cluster]:
        clusters_copy = copy.deepcopy(clusters)
        for cluster in clusters_copy:
            zone_centroids = []
            for zone in cluster.zones:
                coords = pop.coords[zone.indices]
                zone_centroids.append(coords.mean(axis=0))
                if point_strategy is not None:
                    point_sorting = self._sort_points(coords, point_strategy)
                    zone.sort = self._apply_sorting_to_list(zone.sort, point_sorting)
            if zone_strategy is not None:
                zone_centroids = np.array(zone_centroids)
                zone_sorting = self._sort_points(zone_centroids, zone_strategy)
                cluster.zones = self._apply_sorting_to_list(cluster.zones, zone_sorting)
        return clusters_copy

    @staticmethod
    def _sort_points(points: np.ndarray, strategy: str) -> np.ndarray:
        match strategy:
            case 'lexico_xy':
                return np.lexsort((points[:, 1], points[:, 0]))
            case 'lexico_yx':
                return np.lexsort((points[:, 0], points[:, 1]))
            case 'angle':
                angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
                return np.argsort(angles)
            case 'dist_from_origin':
                distances = np.linalg.norm(points, axis=1)
                return np.argsort(distances)
            case 'projection':
                projections = points[:, 0] + points[:, 1]
                return np.argsort(projections)
            case 'dist_from_centroids':
                centroid = points.mean(axis=0)
                distances = np.linalg.norm(points - centroid, axis=1)
                return np.argsort(distances)
            case 'max_coord':
                max_coords = np.max(points, axis=1)
                return np.argsort(max_coords)
            case 'spiral':
                centroid = points.mean(axis=0)
                translated_points = points - centroid
                angles = np.mod(np.arctan2(translated_points[:, 1], translated_points[:, 0]), 2 * np.pi)
                distances = np.linalg.norm(translated_points, axis=1)
                return np.lexsort((distances, angles))
        return np.arange(points.shape[0])

    @staticmethod
    def _apply_sorting_to_list(lst, sorting):
        temp = np.array(lst)
        temp = temp[sorting]
        return temp.tolist()

    def _build_order(self, num_splits: int):
        if not self.clusters:
            return

        indices_list = []
        shares_list = []
        fixed_ids = set()

        for cluster in self.clusters:
            cluster_indices = []
            cluster_shares = []

            # 1. Collect Floor
            if cluster.floor and cluster.floor.index is not None and cluster.floor.index != -1:
                cluster_indices.append(cluster.floor.index)
                cluster_shares.append(cluster.floor.percentage)
                fixed_ids.add(cluster.floor.index)

            # 2. Collect Zones
            for zone in cluster.zones:
                zone_indices = np.repeat(zone.indices, num_splits)
                # Ensure we divide the share correctly by splits
                zone_shares = np.repeat(zone.shares, num_splits) / num_splits
                cluster_indices.extend(zone_indices.tolist())
                cluster_shares.extend(zone_shares.tolist())

            # 3. Collect Ceil
            if cluster.ceil and cluster.ceil.index is not None and cluster.ceil.index != -1:
                cluster_indices.append(cluster.ceil.index)
                cluster_shares.append(cluster.ceil.percentage)
                fixed_ids.add(cluster.ceil.index)

            # --- CRITICAL FIX: Local Normalization ---
            # Every cluster in your 'r' logic should sum to exactly 'r' 
            # (or 1.0 if r=1). This prevents floating point drift.
            c_shares = np.array(cluster_shares)
            target_sum = self.pop.sum_prob(np.array(cluster_indices), c_shares)
            
            # We don't change the indices, but we ensure the 'order' 
            # representation matches the population's expected mass.
            indices_list.extend(cluster_indices)
            shares_list.extend(cluster_shares)

        # self._order = np.column_stack([indices_list, shares_list])
        self._order = np.zeros((len(indices_list), 2))
        self._order[:, 0] = np.array(indices_list, dtype=np.int64)
        self._order[:, 1] = np.array(shares_list, dtype=np.float64)
        # --- ADD THIS LOGIC ---
        total_n = self.pop.n
        current_sum = np.sum(self._order[:, 1] * self.pop.inclusions[self._order[:, 0].astype(int)])
        
        if not np.isclose(current_sum, total_n, atol=1e-7):
            # Scale all shares slightly to match the target sample size n
            correction_factor = total_n / current_sum
            self._order[:, 1] *= correction_factor
        # ----------------------
        
        self._fixed_ids = fixed_ids
        # print("POP SIZE:", len(self.pop.indices))
        # print("ORDER SIZE:", len(indices_list))

        

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
                        valid_zones = [z for z in cluster.zones if len(z.sort) > 1]
                        if not valid_zones:
                            continue
    
                        z_idxs = rng.choice(len(valid_zones), size=min(num_zones, len(valid_zones)), replace=False)
                        selected_zones = [valid_zones[i] for i in z_idxs]
    
                        for zone in selected_zones:
                            for _ in range(num_changes):
                                i, j = rng.choice(len(zone.sort), size=2, replace=False)
                                zone.sort[i], zone.sort[j] = zone.sort[j], zone.sort[i]
                                
    
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
                            i, j = rng.choice(len(cluster.zones), size=2, replace=False)
                            cluster.zones[i], cluster.zones[j] = cluster.zones[j], cluster.zones[i]
    
            # Rebuild the _order array to reflect all new states
            self._build_order(self.num_splits)

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
        # Create a fresh instance. We pass the same pop reference
        # since the pop itself isn't what we are mutating.
        new_instance = Order(self.pop)

        # Copy the numpy array and the set of fixed IDs
        new_instance._order = np.copy(self._order)
        new_instance._fixed_ids = self._fixed_ids.copy()
        new_instance.clusters = copy.deepcopy(self.clusters) if self.clusters is not None else None

        new_instance.num_zones = self.num_zones
        new_instance.num_splits = self.num_splits

        return new_instance
