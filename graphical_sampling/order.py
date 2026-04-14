from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np
from .population import Population

@dataclass
class Zone:
    _shares: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    _indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sort: list[int] = field(default_factory=list)
    virtual_centroid: np.ndarray | None = None

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
        self.pop = population
        self._order: np.ndarray = np.empty((0, 2))
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
        num_splits: int = 1,
        topological: bool = True,
        shift_jump: int = 0,
        shuffle: bool = False
    ) -> Order:
        instance = cls(population)
        num_zones = len(clusters[0].zones)
        
        # Apply spatial sorting and the Phase Shift (Jump) logic
        instance.clusters = instance._modify_clusters(
            population, clusters, zone_strategy, point_strategy, topological, shift_jump, shuffle
        )

        instance._build_order(num_splits)
        instance.num_zones = num_zones
        instance.num_splits = num_splits
        return instance

    def _modify_clusters(self, population, clusters, zone_strategy, point_strategy, topological, shift_jump, shuffle):
        new_clusters = []
        for c_idx, cluster in enumerate(clusters):
            # 1. Calculate Zone Centroids
            zone_coords = []
            for z in cluster.zones:
                if len(z.indices) > 0:
                    zone_coords.append(population.coords[z.indices].mean(axis=0))
                else:
                    zone_coords.append(np.array([0.5, 0.5]))
            zone_centroids = np.array(zone_coords)

            # 2. Get Base Order (Stable base for jumping)
            strategy = zone_strategy if zone_strategy is not None else 'lexico_yx'
            base_order = self._get_strategy_order(zone_centroids, strategy).tolist()
            
            # 3. Apply the Phase Shift
            num_z = len(base_order)
            jump_val = (c_idx * 7) % num_z if shuffle else (c_idx * shift_jump) % num_z
            shifted_order = base_order[jump_val:] + base_order[:jump_val]
            
            # Reconstruct zones list for this cluster
            modified_zones = [copy.deepcopy(cluster.zones[idx]) for idx in shifted_order]

            # 4. Apply Point-Level strategy within each zone
            if point_strategy is not None:
                for zone in modified_zones:
                    if len(zone.indices) > 1:
                        pts = population.coords[zone.indices]
                        zone.sort = self._get_strategy_order(pts, point_strategy).tolist()
                    else:
                        zone.sort = [0] if len(zone.indices) == 1 else []

            new_cluster = copy.copy(cluster)
            new_cluster.zones = modified_zones
            new_clusters.append(new_cluster)
        return new_clusters

    def _get_strategy_order(self, points: np.ndarray, strategy: str) -> np.ndarray:
        if points.shape[0] == 0: return np.array([], dtype=int)
        if points.ndim == 1: points = points.reshape(1, -1)
            
        match strategy:
            case 'lexico_xy': return np.lexsort((points[:, 1], points[:, 0]))
            case 'lexico_yx': return np.lexsort((points[:, 0], points[:, 1]))
            case 'projection': return np.argsort(points[:, 0] + points[:, 1])
            case 'dist_from_origin': return np.argsort(np.linalg.norm(points, axis=1))
            case 'angle':
                center = points.mean(axis=0)
                return np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
            case 'spiral':
                center = points.mean(axis=0)
                dists = np.linalg.norm(points - center, axis=1)
                angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
                return np.lexsort((angles, dists))
            case _: return np.arange(len(points))

    def _build_order(self, num_splits: int):
        indices_list, shares_list = [], []
        fixed_ids = set()

        for cluster in self.clusters:
            if cluster.floor and cluster.floor.index is not None and cluster.floor.index != -1:
                indices_list.append(cluster.floor.index)
                shares_list.append(cluster.floor.percentage)
                fixed_ids.add(cluster.floor.index)

            for zone in cluster.zones:
                if len(zone.indices) > 0:
                    indices_list.extend(np.repeat(zone.indices, num_splits).tolist())
                    shares_list.extend((np.repeat(zone.shares, num_splits) / num_splits).tolist())

            if cluster.ceil and cluster.ceil.index is not None and cluster.ceil.index != -1:
                indices_list.append(cluster.ceil.index)
                shares_list.append(cluster.ceil.percentage)
                fixed_ids.add(cluster.ceil.index)

        self._order = np.column_stack([np.array(indices_list, dtype=int), np.array(shares_list, dtype=float)])
        
        # Final Quota Normalization to ensure exactly 'n' units are sampled
        total_n = self.pop.n
        current_sum = np.sum(self._order[:, 1] * self.pop.inclusions[self._order[:, 0].astype(int)])
        if not np.isclose(current_sum, total_n, atol=1e-8):
            self._order[:, 1] *= (total_n / current_sum)
        
        self._fixed_ids = fixed_ids

    def change(self, num_clusters: int, num_zones: int, num_changes: int, num_zone_changes: int):
        rng = np.random.default_rng()
        if num_changes > 0:
            c_idxs = rng.choice(len(self.clusters), size=min(num_clusters, len(self.clusters)), replace=False)
            for idx in c_idxs:
                cluster = self.clusters[idx]
                z_idxs = rng.choice(len(cluster.zones), size=min(num_zones, len(cluster.zones)), replace=False)
                for z_idx in z_idxs:
                    zone = cluster.zones[z_idx]
                    if len(zone.sort) > 1:
                        for _ in range(num_changes):
                            i, j = rng.choice(len(zone.sort), size=2, replace=False)
                            zone.sort[i], zone.sort[j] = zone.sort[j], zone.sort[i]

        if num_zone_changes > 0:
            c_idxs = rng.choice(len(self.clusters), size=min(num_clusters, len(self.clusters)), replace=False)
            for idx in c_idxs:
                cluster = self.clusters[idx]
                if len(cluster.zones) > 1:
                    for _ in range(num_zone_changes):
                        i, j = rng.choice(len(cluster.zones), size=2, replace=False)
                        cluster.zones[i], cluster.zones[j] = cluster.zones[j], cluster.zones[i]
        
        self._build_order(self.num_splits)

    def get(self) -> np.ndarray:
        return self._order

    def copy(self) -> Order:
        new_instance = Order(self.pop)
        new_instance._order = np.copy(self._order)
        new_instance._fixed_ids = self._fixed_ids.copy()
        new_instance.clusters = copy.deepcopy(self.clusters)
        new_instance.num_zones = self.num_zones
        new_instance.num_splits = self.num_splits
        return new_instance

    @property
    def fixed_ids(self) -> set[int]:
        return self._fixed_ids
