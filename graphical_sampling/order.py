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
        
        # --- IMPROVEMENT 1: ASPECT-RATIO-AWARE GRID DIMENSIONS ---
        # Instead of a square root, we use the actual span of the coordinates
        if grid_dims is not None:
            num_cols, num_rows = grid_dims
        else:
            span = np.ptp(points, axis=0)
            ratio = span[0] / (span[1] + 1e-12)
            num_rows = int(np.sqrt(points.shape[0] / (ratio + 1e-12)))
            num_rows = max(1, num_rows)
            num_cols = points.shape[0] // num_rows

        # --- IMPROVEMENT 2: COORDINATE JITTER TO BREAK TIES ---
        # Prevents deterministic Lexico artifacts in dense topological bins
        pts = points.copy()
        if topological:
            pts += np.random.uniform(-1e-9, 1e-9, size=pts.shape)
            
            # Re-binning with jittered coordinates
            y_indices = np.argsort(pts[:, 1])
            for row_id in range(num_rows):
                start, end = row_id * num_cols, (row_id + 1) * num_cols
                pts[y_indices[start:end], 1] = row_id 
                
            x_indices = np.argsort(pts[:, 0])
            for col_id in range(num_cols):
                start, end = col_id * num_rows, (col_id + 1) * num_rows
                pts[x_indices[start:end], 0] = col_id
            
        match strategy:
            # --- IMPROVEMENT 3: HILBERT SPACE-FILLING CURVE ---
            # Superior to Snake/Lexico for preserving 2D locality in 1D
            case 'hilbert':
                def d2xy(n, d):
                    """Converts 1D Hilbert index to 2D coordinates."""
                    t = d
                    x = y = 0
                    s = 1
                    while s < n:
                        rx = 1 & (t // 2)
                        ry = 1 & (t ^ rx)
                        # Rotate/Flip logic
                        if ry == 0:
                            if rx == 1:
                                x, y = s - 1 - x, s - 1 - y
                            x, y = y, x
                        x += s * rx
                        y += s * ry
                        t //= 4
                        s *= 2
                    return x, y

                def xy2d(n, x, y):
                    """Converts 2D coordinates to 1D Hilbert index."""
                    d = 0
                    s = n // 2
                    while s > 0:
                        rx = (x & s) > 0
                        ry = (y & s) > 0
                        d += s * s * ((3 * rx) ^ ry)
                        # Rotate/Flip logic
                        if ry == 0:
                            if rx == 1:
                                x, y = s - 1 - x, s - 1 - y
                            x, y = y, x
                        s //= 2
                    return d

                # Find smallest power of 2 that covers the grid
                n_hilbert = 2**int(np.ceil(np.log2(max(num_cols, num_rows))))
                hilbert_indices = []
                for i in range(pts.shape[0]):
                    # Map topological bin to Hilbert index
                    h_idx = xy2d(n_hilbert, int(pts[i, 0]), int(pts[i, 1]))
                    hilbert_indices.append(h_idx)
                return np.argsort(hilbert_indices)

            # --- IMPROVEMENT 4: REFINED SNAKE (Aspect-Ratio Aware) ---
            case 'snake_refined':
                y_coords = pts[:, 1]
                unique_y = np.unique(y_coords)
                base_idx = np.lexsort((pts[:, 0], pts[:, 1]))
                
                final_order = []
                for i, y in enumerate(unique_y):
                    row_indices = base_idx[pts[base_idx, 1] == y]
                    if i % 2 == 1: # Reverse every odd row to "snake"
                        row_indices = row_indices[::-1]
                    final_order.extend(row_indices.tolist())
                return np.array(final_order)
            case 'hilbert_strong':
                def rot(n, x, y, rx, ry):
                    if ry == 0:
                        if rx == 1:
                            x, y = n-1 - x, n-1 - y
                        return y, x
                    return x, y

                def xy2d(n, x, y):
                    d = 0
                    s = n // 2
                    while s > 0:
                        rx = 1 if (x & s) > 0 else 0
                        ry = 1 if (y & s) > 0 else 0
                        d += s * s * ((3 * rx) ^ ry)
                        x, y = rot(s, x, y, rx, ry)
                        s //= 2
                    return d

                n_size = 2**int(np.ceil(np.log2(max(num_cols, num_rows))))
                # Apply a random flip/rotation seed to the coordinate space
                seed = np.random.randint(0, 4)
                hilbert_indices = []
                for i in range(pts.shape[0]):
                    ix, iy = int(pts[i, 0]), int(pts[i, 1])
                    if seed == 1: ix, iy = iy, ix
                    elif seed == 2: ix = n_size - 1 - ix
                    h_idx = xy2d(n_size, ix, iy)
                    hilbert_indices.append(h_idx)
                return np.argsort(hilbert_indices)

            # --- STRENGTH 2: MORTON WITH XOR MASK (GRTS STYLE) ---
            # This is exactly how GRTS achieves spatial balance: bit-interleaving + randomization
            case 'morton_grts':
                def part1by1(x):
                    x &= 0x0000ffff
                    x = (x | (x << 8)) & 0x00ff00ff
                    x = (x | (x << 4)) & 0x0f0f0f0f
                    x = (x | (x << 2)) & 0x33333333
                    x = (x | (x << 1)) & 0x55555555
                    return x

                xor_mask = np.random.randint(0, 0xFFFFFFFF)
                morton_indices = []
                for i in range(pts.shape[0]):
                    # Interleave bits of X and Y
                    m_idx = (part1by1(int(pts[i, 1])) << 1) | part1by1(int(pts[i, 0]))
                    # XOR mask breaks deterministic clustering
                    morton_indices.append(m_idx ^ xor_mask)
                return np.argsort(morton_indices)

            # --- STRENGTH 3: STAGGERED SNAKE (MAXIMIZES BOUNDARY SPREAD) ---
            # Standard Snake has weak return-jumps; Staggered Snake offsets rows
            case 'snake_staggered':
                y_coords = pts[:, 1]
                unique_y = np.unique(y_coords)
                final_order = []
                for i, y in enumerate(unique_y):
                    row_indices = np.where(pts[:, 1] == y)[0]
                    # Sort row by X
                    row_indices = row_indices[np.argsort(pts[row_indices, 0])]
                    # Every second row, apply a circular shift of half the row length
                    if i % 2 == 1:
                        row_indices = row_indices[::-1]
                        shift = len(row_indices) // 2
                        row_indices = np.roll(row_indices, shift)
                    final_order.extend(row_indices.tolist())
                return np.array(final_order)
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
                # print(f"Final Path (1-indexed): {np.array(final_order) + 1}")
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
        indices_list = []
        fixed_ids = set()

        for cluster in self.clusters:
            if cluster.floor and cluster.floor.index is not None and cluster.floor.index != -1:
                indices_list.append(cluster.floor.index)
                fixed_ids.add(cluster.floor.index)

            for zone in cluster.zones:
                if len(zone.indices) > 0:
                    # We don't care about shares here, just get the spatial sequence
                    indices_list.extend(zone.indices)

            if cluster.ceil and cluster.ceil.index is not None and cluster.ceil.index != -1:
                indices_list.append(cluster.ceil.index)
                fixed_ids.add(cluster.ceil.index)

        raw_ids = np.array(indices_list, dtype=int)
        
        # =====================================================================
        # THE CONTIGUOUS COLLAPSE
        # =====================================================================
        # 1. Find the FIRST appearance of every unit to keep the spatial path intact
        _, idx_first_appearance = np.unique(raw_ids, return_index=True)
        appearance_order = np.sort(idx_first_appearance)
        final_ids = raw_ids[appearance_order]
        
        # 2. Catch any units totally dropped by the 9x9 float rounding
        missing_ids = np.setdiff1d(np.arange(self.pop.N), final_ids)
        if len(missing_ids) > 0:
            final_ids = np.concatenate([final_ids, missing_ids])
            
        # 3. Handle num_splits cleanly without shattering units across the map
        if num_splits > 1:
            final_ids = np.tile(final_ids, num_splits)
            final_shares = np.full_like(final_ids, 1.0 / num_splits, dtype=float)
        else:
            final_shares = np.ones_like(final_ids, dtype=float)

        self._order = np.column_stack([final_ids, final_shares])
        self._fixed_ids = fixed_ids

    def change(self, num_clusters: int, num_zones: int, num_changes: int, num_zone_changes: int, window: int | None = None):
        rng = np.random.default_rng()
        
        if num_changes > 0:
            c_idxs = rng.choice(len(self.clusters), size=min(num_clusters, len(self.clusters)), replace=False)
            for idx in c_idxs:
                cluster = self.clusters[idx]
                z_idxs = rng.choice(len(cluster.zones), size=min(num_zones, len(cluster.zones)), replace=False)
                
                for z_idx in z_idxs:
                    zone = cluster.zones[z_idx]
                    n = len(zone.sort)
                    
                    if n > 1:
                        for _ in range(num_changes):
                            # =========================================================
                            # FAST PATH: ORIGINAL BEHAVIOR (Sledgehammer)
                            # =========================================================
                            if window is None:
                                # This is the native, C-optimized NumPy call
                                i, j = rng.choice(n, size=2, replace=False)
                                zone.sort[i], zone.sort[j] = zone.sort[j], zone.sort[i]
                            
                            # =========================================================
                            # PRECISION PATH: WINDOWED POLISHING (Scalpel)
                            # =========================================================
                            else:
                                i = rng.integers(0, n)
                                low = max(0, i - window)
                                high = min(n, i + window + 1)
                                
                                # This list comprehension is slightly slower, 
                                # but we only use it during the final global polish
                                choices = [idx for idx in range(low, high) if idx != i]
                                if choices:
                                    j = rng.choice(choices)
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
