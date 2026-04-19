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
        shuffle: bool = False,
        grid_dims: tuple[int, int] | None = None
    ) -> Order:
        instance = cls(population)
        num_zones = len(clusters[0].zones)
        
        # Apply spatial sorting and the Phase Shift (Jump) logic
        instance.clusters = instance._modify_clusters(
            population, clusters, zone_strategy, point_strategy, topological, shift_jump, shuffle, grid_dims
        )

        instance._build_order(num_splits)
        instance.num_zones = num_zones
        instance.num_splits = num_splits
        return instance

    def _modify_clusters(self, population, clusters, zone_strategy, point_strategy, topological, shift_jump, shuffle, grid_dims=None):
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

            # 2. Zone-Level Sorting
            strategy = zone_strategy if zone_strategy is not None else 'lexico_yx'
            # PASS grid_dims HERE
            base_order = self._get_strategy_order(zone_centroids, strategy, topological=topological, grid_dims=grid_dims).tolist()
            
            # 3. Apply the Phase Shift (Jump)
            num_z = len(base_order)
            jump_val = (c_idx * 7) % num_z if shuffle else (c_idx * shift_jump) % num_z
            shifted_order = base_order[jump_val:] + base_order[:jump_val]
            
            modified_zones = [copy.deepcopy(cluster.zones[idx]) for idx in shifted_order]

            # 4. Point-Level Sorting
            if point_strategy is not None:
                for zone in modified_zones:
                    if len(zone.indices) > 1:
                        pts_coords = population.coords[zone.indices]
                        # Force topological=False for points
                        zone.sort = self._get_strategy_order(pts_coords, point_strategy, topological=False).tolist()
                    else:
                        zone.sort = [0] if len(zone.indices) == 1 else []

            new_cluster = copy.copy(cluster)
            new_cluster.zones = modified_zones
            new_clusters.append(new_cluster)
        return new_clusters

    def _get_strategy_order(self, points, strategy, topological=False, grid_dims=None):
        if points.shape[0] == 0: return np.array([], dtype=int)
        
        # Determine Grid Dimensions
        if grid_dims is not None:
            num_cols, num_rows = grid_dims
        else:
            num_rows = int(np.sqrt(points.shape[0]))
            num_cols = points.shape[0] // num_rows

        pts = points.copy()
        if topological:
            # 1. Group into exactly num_rows
            y_indices = np.argsort(points[:, 1])
            for row_id in range(num_rows):
                start = row_id * num_cols
                end = (row_id + 1) * num_cols
                pts[y_indices[start:end], 1] = row_id 
                
            # 2. Group into exactly num_cols
            x_indices = np.argsort(points[:, 0])
            for col_id in range(num_cols):
                start = col_id * num_rows
                end = (col_id + 1) * num_rows
                pts[x_indices[start:end], 0] = col_id
            
        match strategy:
            case 'knight_move':
                num_pts = pts.shape[0]
                # Determine grid dimensions
                if grid_dims is not None:
                    num_cols, num_rows = grid_dims
                else:
                    num_rows = int(np.sqrt(num_pts))
                    num_cols = num_pts // num_rows

                # 1. Snap centroids to a clean grid to avoid float jitters
                # We use the binned coordinates from the topological step
                x_indices = np.argsort(pts[:, 0])
                cols = np.array_split(x_indices, num_cols)
                
                # Re-map points to their "Column ID" and "Row ID" for logical sorting
                logical_grid = np.zeros((num_pts, 2), dtype=int)
                for c_id, col_indices in enumerate(cols):
                    # Within each column, sort by Y to assign Row IDs
                    row_sort = np.argsort(pts[col_indices, 1])
                    for r_id, idx in enumerate(col_indices[row_sort]):
                        logical_grid[idx] = [c_id, r_id]

                final_order = []

                # 2. Iterate through Column Pairs (0-1, 2-3, 4-5)
                for pair_start in range(0, num_cols, 2):
                    c1, c2 = pair_start, pair_start + 1
                    
                    # 3. Determine vertical direction for this column pair
                    # Pair 0 (Left): Bottom to Top | Pair 1 (Middle): Top to Bottom | Pair 2: Bottom to Top
                    is_descending = (pair_start // 2) % 2 == 1
                    row_range = range(num_rows - 1, -1, -2) if is_descending else range(0, num_rows, 2)

                    for r_start in row_range:
                        # We are looking at a 2x2 square block
                        # logical coords: (c1, r_start), (c2, r_start), (c1, r_start+1), (c2, r_start+1)
                        # We want the path: 1 (BL) -> 2 (BR) -> 4 (TR) -> 3 (TL)
                        
                        # Find indices matching these logical coordinates
                        # Pattern: [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
                        square_coords = [
                            (c1, r_start), (c2, r_start), 
                            (c2, r_start + 1), (c1, r_start + 1)
                        ] if not is_descending else [
                            (c1, r_start), (c2, r_start),
                            (c2, r_start - 1), (c1, r_start - 1)
                        ]

                        for target_c, target_r in square_coords:
                            # Find the original index that was assigned this logical (col, row)
                            for idx in range(num_pts):
                                if logical_grid[idx, 0] == target_c and logical_grid[idx, 1] == target_r:
                                    final_order.append(idx)
                                    break

                # Verification output
                print(f"Final Path (1-indexed): {np.array(final_order) + 1}")
                return np.array(final_order)
                
            
            case 'snake':
                # Use binned coordinates if topological, otherwise raw
                y_coords = pts[:, 1]
                unique_y = np.unique(y_coords)
                # Sort by Y then X
                base_idx = np.lexsort((pts[:, 0], pts[:, 1]))
                
                final_order = []
                for i, y in enumerate(unique_y):
                    row_indices = base_idx[pts[base_idx, 1] == y]
                    if i % 2 == 1: # Reverse every odd row
                        row_indices = row_indices[::-1]
                    final_order.extend(row_indices.tolist())
                return np.array(final_order)

            case 'lexico_xy': 
                return np.lexsort((pts[:, 1], pts[:, 0]))

            case 'lexico_yx': 
                return np.lexsort((pts[:, 0], pts[:, 1]))
        

            case 'lexico_xy': 
                # Primary sort on X, secondary on Y
                # print(np.lexsort((pts[:, 1], pts[:, 0])))
                return np.lexsort((pts[:, 1], pts[:, 0]))

            case 'lexico_yx': 
                # Primary sort on Y, secondary on X
                return np.lexsort((pts[:, 0], pts[:, 1]))

            case 'projection': 
                return np.argsort(pts[:, 0] + pts[:, 1])

            case 'dist_from_origin': 
                return np.argsort(np.linalg.norm(pts, axis=1))

            case 'angle':
                center = pts.mean(axis=0)
                return np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))

            case 'spiral':
                center = pts.mean(axis=0)
                dists = np.linalg.norm(pts - center, axis=1)
                angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
                return np.lexsort((angles, dists))
            case 'pure_random':
                # Completely random selection within the zone
                rng = np.random.default_rng()
                return rng.permutation(len(points))
            case 'none_baseline':
                # Returns indices in their original, un-sorted state
                return np.arange(len(points))
            case 'stratified_shuffle':
                indices = np.arange(len(points))
                np.random.shuffle(indices)
                return indices

            case _: 
                return np.arange(len(pts))

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
