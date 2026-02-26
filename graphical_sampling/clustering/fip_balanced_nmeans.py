import warnings
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from numba import jit

from ..population import Population
from ..order import Cluster, Zone, Floor, Ceil


@jit(nopython=True)
def _greedy_jitter_numba(D, n, start_node):
    """
    Greedy Nearest Neighbor with minimal probabilistic noise (Jitter).
    """
    path = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)

    curr = start_node
    path[0] = curr
    visited[curr] = True

    for i in range(1, n):
        # Copy row to avoid modifying original matrix
        # Note: In Numba, row slicing D[curr] is a view, so we copy.
        dists = D[curr].copy()

        # Mark visited as infinity
        for v_idx in range(n):
            if visited[v_idx]:
                dists[v_idx] = np.inf

        # PROBABILISTIC TRICK:
        # Instead of sorting to find top 3 (slow), we multiply unvisited
        # distances by a random factor between 0.9 and 1.1.
        # This sometimes makes the 2nd or 3rd closest look like the closest.
        # It preserves the "structure" of the data while adding variety.
        noise = np.random.rand(n) * 0.2 + 0.9 # Range [0.9, 1.1]
        dists *= noise

        nxt = np.argmin(dists)

        path[i] = nxt
        visited[nxt] = True
        curr = nxt

    return path


@jit(nopython=True)
def _two_opt_open_numba(D, path, n):
    """
    Optimizes Open Path.
    Can flip ANY segment, including the start and end nodes relative to the path structure.
    (Note: In Open TSP, rotating the array doesn't change length, but reversing inner segments does).
    """
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if D[path[i-1], path[j]] + D[path[i], path[j+1]] < \
                   D[path[i-1], path[i]] + D[path[j], path[j+1]]:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True


class OpenTSPSolver:
    def solve(self, points: np.ndarray, restarts: int = 15) -> np.ndarray:
        """
        Finds the best OPEN path (Start -> End) by trying multiple random start nodes.

        Args:
            points: (N, 2) array of coordinates.
            restarts: Number of random start nodes to try.
                      Higher = better quality, linearly more time.
                      For N < 1000, 20 restarts is near-instant.
        """
        n = len(points)
        if n < 3:
            return np.arange(n)

        # For very large N, fallback to KD-Tree (Safe Mode)
        if n >= 5000:
            return self._solve_large_safe(points, n)

        # For N < 5000: Use Numba-accelerated Multi-Start
        return self._solve_numba_accelerated(points, n, restarts)

    def _solve_numba_accelerated(self, points, n, restarts):
        # 1. Precompute Distance Matrix (Heavy lifting done once)
        D = squareform(pdist(points))

        # 2. Run the heavy logic in Numba
        # We pass the matrix and let the JIT function handle the looping.
        best_path = self._run_multi_start_numba(D, n, restarts)

        return best_path

    @staticmethod
    @jit(nopython=True)
    def _run_multi_start_numba(D, n, restarts):
        """
        Runs the full pipeline (Greedy + 2-Opt) 'restarts' times.
        Fully Compiled = Blazing Fast.
        """
        best_path = np.empty(n, dtype=np.int32) # Placeholder
        best_dist = np.inf

        # We will try different random starts
        for i in range(restarts):

            # --- 1. Randomized Start & Construction ---
            # Pick a random start node
            start_node = np.random.randint(0, n)

            # Build path with "Jitter" (Probabilistic Greedy)
            # We add slight noise to distances to simulate picking from "top k"
            # without the expensive sorting cost.
            current_path = _greedy_jitter_numba(D, n, start_node)

            # --- 2. 2-Opt Optimization ---
            _two_opt_open_numba(D, current_path, n)

            # --- 3. Evaluate ---
            current_dist = 0.0
            for k in range(n - 1):
                current_dist += D[current_path[k], current_path[k+1]]

            if current_dist < best_dist:
                best_dist = current_dist
                best_path = current_path.copy()

        return best_path

    @staticmethod
    def _solve_large_safe(points, n):
        # (Legacy KD-Tree implementation for N > 5000)
        tree = cKDTree(points)
        visited = np.zeros(n, dtype=bool)
        path = np.zeros(n, dtype=np.int32)
        current_idx = 0
        path[0] = 0
        visited[0] = True
        k_search = 10
        for i in range(1, n):
            found = False
            while not found:
                dists, indices = tree.query(points[current_idx], k=min(n, k_search))
                if np.ndim(indices) == 0:
                    indices = [indices]
                for idx in indices:
                    if not visited[idx]:
                        current_idx = idx
                        found = True
                        break
                if not found:
                    k_search *= 2
                    if k_search >= n:
                        remaining = np.where(~visited)[0]
                        if len(remaining) > 0:
                            current_idx = remaining[0]
                            found = True
                        else:
                            break
            path[i] = current_idx
            visited[current_idx] = True
            if k_search > 50:
                k_search = 10
        return path


class FIPBalancedNMeans:
    def __init__(self, n: int, n_init=50, tol=1e-9, max_iter=100) -> None:
        self.n = n
        self.population: Population | None = None
        self.labels: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.path_order: np.ndarray | None = None
        self.membership: np.ndarray | None = None
        self.clusters: list[Cluster] | None = None
        self.tsp_solver = OpenTSPSolver()

        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, population: Population, init_centroids: np.ndarray | None = None) -> None:
        self.population = population
        coords = population.coords
        probs = population.inclusions
        N = len(coords)

        # 1. Initial Clustering
        raw_labels, raw_centroids = self._get_labels_centroids(coords, probs, init_centroids)

        # 2. TSP Ordering
        # returns a linear path [Start -> ... -> End]
        self.path_order = self.tsp_solver.solve(raw_centroids)

        # 3. Exact Balanced Clusters
        self.clusters = self._generate_exact_clusters(
            self.path_order, raw_labels, coords, probs, population.indices
        )

        # 4. Finalize Outputs
        self.membership = self._generate_membership(self.clusters, N, population.indices)
        self.labels = np.argmax(self.membership, axis=1)

        # Recalculate centroids based on final hard labels
        self.centroids = np.array([
            coords[self.labels == i].mean(axis=0) if np.any(self.labels == i)
            else np.zeros(coords.shape[1]) for i in range(self.n)
        ])

    def fit_zones(self, num_zones: int) -> None:
        """
        Splits each existing cluster's population into `num_zones` sub-zones.
        The borders (floor/ceil) of the parent cluster remain untouched.
        """
        if self.clusters is None:
            raise ValueError("The main clusters have not been fitted yet. Call .fit() first.")

        for i, cluster in enumerate(self.clusters):
            current_ids = cluster.zones[0].ids

            # 1. Safely gather indices and shares, accounting for potential -1 (no border)
            subset_indices = []
            shares = []

            has_floor = cluster.floor.index != -1
            has_ceil = cluster.ceil.index != -1

            if has_floor:
                subset_indices.append(cluster.floor.index)
                shares.append(cluster.floor.percentage)

            subset_indices.extend(current_ids)
            shares.extend([1.0] * len(current_ids))

            if has_ceil:
                subset_indices.append(cluster.ceil.index)
                shares.append(cluster.ceil.percentage)

            subset_indices = np.array(subset_indices, dtype=int)
            shares = np.array(shares, dtype=float)

            # 2. Create subpopulation subset using your sharing logic
            sp = self.population.subset(subset_indices, share=shares)

            # 3. Fit the subset to get `num_zones` sub-clusters
            sub_fbn = FIPBalancedNMeans(
                n=num_zones,
                n_init=self.n_init,
                tol=self.tol,
                max_iter=self.max_iter
            )
            sub_fbn.fit(sp)

            # 4. Extract the new zones, filtering out the parent's floor/ceil
            new_zones = []
            for j, sc in enumerate(sub_fbn.clusters):
                sc_indices = []

                # Gather all indices from the sub-cluster (its zones, floor, and ceil)
                for z in sc.zones:
                    sc_indices.extend(z.ids)
                if sc.floor.index != -1:
                    sc_indices.append(sc.floor.index)
                if sc.ceil.index != -1:
                    sc_indices.append(sc.ceil.index)

                # Filter out the parent's border indices to keep them isolated
                filtered_ids = []
                seen = set()
                for idx in sc_indices:
                    if idx not in seen:
                        seen.add(idx)
                        # Only keep the index if it IS NOT the parent's floor or ceil
                        if (not has_floor or idx != cluster.floor.index) and \
                                (not has_ceil or idx != cluster.ceil.index):
                            filtered_ids.append(idx)

                new_zones.append(Zone(label=j, ids=filtered_ids))

            # 5. Replace the parent's single zone with the new partitioned zones
            self.clusters[i].zones = new_zones

    def _get_labels_centroids(
            self,
            coords: np.ndarray,
            probs: np.ndarray,
            init_centroids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*divide by zero.*")
            warnings.filterwarnings("ignore", message=".*overflow.*")
            warnings.filterwarnings("ignore", message=".*invalid value.*")

            probs_normalized = probs / probs.sum() * len(coords)

            if init_centroids is not None:
                kmeans = KMeans(
                    n_clusters=self.n,
                    init=init_centroids,
                    tol=self.tol,
                    max_iter=self.max_iter
                )
                labels = kmeans.fit_predict(coords, sample_weight=probs_normalized)
                centroids = kmeans.cluster_centers_
                return labels, centroids

            best_error = np.inf
            best_labels = None
            best_centroids = None

            for _ in range(self.n_init):
                kmeans = KMeans(n_clusters=self.n, n_init=1, tol=self.tol, max_iter=self.max_iter)
                raw_labels = kmeans.fit_predict(coords, sample_weight=probs_normalized)
                raw_centroids = kmeans.cluster_centers_

                sums = np.array([probs[raw_labels == i].sum() for i in range(self.n)])
                mean_abs_error = np.abs(sums - 1).sum()

                if mean_abs_error < best_error:
                    best_error = mean_abs_error
                    best_labels = raw_labels
                    best_centroids = raw_centroids

            return best_labels, best_centroids

    def _generate_exact_clusters(
            self, order: np.ndarray, labels: np.ndarray,
            coords: np.ndarray, probs: np.ndarray, pop_indices: np.ndarray
    ) -> list[Cluster]:

        # Group indices by label
        clusters_idx = {lab: np.flatnonzero(labels == lab) for lab in order}
        mega_indices = []

        # Inter-Cluster Sorting
        for i, lab in enumerate(order):
            curr_idx = clusters_idx.get(lab, np.array([], dtype=int))
            if curr_idx.size == 0: continue

            key = np.zeros(len(curr_idx))

            # Distance to PREV cluster
            if i > 0:
                prev_idx = clusters_idx.get(order[i - 1], np.array([], dtype=int))
                if prev_idx.size > 0:
                    d_prev = cdist(coords[curr_idx], coords[prev_idx]).min(axis=1)
                    key += d_prev

            # Distance to NEXT cluster
            if i < self.n - 1:
                next_idx = clusters_idx.get(order[i + 1], np.array([], dtype=int))
                if next_idx.size > 0:
                    d_next = cdist(coords[curr_idx], coords[next_idx]).min(axis=1)
                    key -= d_next

            # Sort locally: low key -> close to prev; high key -> close to next
            mega_indices.append(curr_idx[np.argsort(key)])

        if not mega_indices:
            return []
        full_indices = np.concatenate(mega_indices)

        # Exact Mass Splitting (Quota)
        ordered_probs = probs[full_indices]
        cum_probs = np.cumsum(ordered_probs)
        total_mass = cum_probs[-1]
        target_mass = total_mass / self.n
        thresholds = np.arange(1, self.n) * target_mass

        # Find split points
        split_indices = np.array(np.searchsorted(cum_probs, thresholds, side='left'))

        final_clusters = []
        start_idx = 0
        split_idx = -1
        prev_border_remainder = 0.0
        prev_border_idx = -1

        for i in range(self.n):
            # Determine boundaries and fractional border ownership
            if i < self.n - 1:
                split_idx = split_indices[i]

                mass_at_split = cum_probs[split_idx]
                mass_before = mass_at_split - ordered_probs[split_idx]
                needed = thresholds[i] - mass_before
                available = ordered_probs[split_idx]

                frac_curr = np.clip(needed / available, 0.0, 1.0)

                curr_border_idx = full_indices[split_idx]
                end_idx = split_idx
            else:
                end_idx = len(full_indices)
                frac_curr = 0.0
                curr_border_idx = -1

            # Extract free indices
            free_indices = full_indices[start_idx: end_idx]

            # Convert internal indices to population indices
            free_ids = pop_indices[free_indices] if pop_indices is not None else free_indices
            floor_id = pop_indices[prev_border_idx] if (
                        pop_indices is not None and prev_border_idx != -1) else prev_border_idx
            ceil_id = pop_indices[curr_border_idx] if (
                        pop_indices is not None and curr_border_idx != -1) else curr_border_idx

            final_clusters.append(
                Cluster(
                    label=int(order[i]),
                    zones=[Zone(label=0, ids=free_ids.tolist())],
                    floor=Floor(index=int(floor_id), percentage=float(prev_border_remainder)),
                    ceil=Ceil(index=int(ceil_id), percentage=float(frac_curr))
                )
            )

            # Setup for next iteration
            start_idx = split_idx + 1
            if curr_border_idx != -1:
                prev_border_idx = curr_border_idx
                prev_border_remainder = 1.0 - frac_curr
                # If the split consumed the point exactly, reset
                if np.isclose(frac_curr, 1.0):
                    prev_border_remainder = 0.0
                    prev_border_idx = -1
            else:
                prev_border_idx = -1
                prev_border_remainder = 0.0

        return final_clusters

    def _generate_membership(self, clusters: list[Cluster], N: int,
                             pop_indices: np.ndarray | None = None) -> np.ndarray:
        membership = np.zeros((N, self.n), dtype=float)

        # If we are working on a subset, the clusters store global indices.
        # We must reverse-map them to local indices (0 to N-1) to safely build this array.
        if pop_indices is not None:
            g2l = {global_idx: local_idx for local_idx, global_idx in enumerate(pop_indices)}

            for cluster in clusters:
                for zone in cluster.zones:
                    if len(zone.ids) > 0:
                        local_ids = [g2l[i] for i in zone.ids]
                        membership[local_ids, cluster.label] = 1.0
                if cluster.floor.index != -1 and cluster.floor.percentage > 1e-9:
                    membership[g2l[cluster.floor.index], cluster.label] += cluster.floor.percentage
                if cluster.ceil.index != -1 and cluster.ceil.percentage > 1e-9:
                    membership[g2l[cluster.ceil.index], cluster.label] += cluster.ceil.percentage
        else:
            # Standard execution for the main population (no mapping needed)
            for cluster in clusters:
                for zone in cluster.zones:
                    if len(zone.ids) > 0:
                        membership[zone.ids, cluster.label] = 1.0
                if cluster.floor.index != -1 and cluster.floor.percentage > 1e-9:
                    membership[cluster.floor.index, cluster.label] += cluster.floor.percentage
                if cluster.ceil.index != -1 and cluster.ceil.percentage > 1e-9:
                    membership[cluster.ceil.index, cluster.label] += cluster.ceil.percentage

        return membership

    def plot(
            self,
            mode: Literal['soft', 'hard'],
            ax: plt.Axes | None = None,
            background_gdf=None,
            show_centroids: bool = False,
            connect_centroids: bool = False,
            size_scale: float = 1000.0,
            figsize: tuple[int, int] = (8, 6),
            dpi: int = 100,
            # Added: control over zone plotting when fitting_zones is active
            show_zone_sub_hulls: bool = True,
            show_zone_labels: bool = True
    ) -> plt.Axes:

        def _draw_hull(points: np.ndarray, color: str, alpha: float, edge_color: str, lw: float):
            if points.shape[0] < 3:
                return None
            try:
                hull = ConvexHull(points)
                vertices = points[hull.vertices]
            except (QhullError, ValueError):
                return None
            ax.add_patch(Polygon(vertices, closed=True, facecolor=color, alpha=alpha,
                                 edgecolor=edge_color, lw=lw, zorder=1))
            return vertices.mean(axis=0)

        def _draw_zone_centroid_label(points: np.ndarray, label: int):
            if points.shape[0] == 0:
                return
            # Use geometric mean of all points in the zone
            centroid = points.mean(axis=0)
            ax.text(centroid[0], centroid[1], str(label+1),
                    size=20, weight='bold', color='black',
                    horizontalalignment='center', verticalalignment='center', zorder=10)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if background_gdf is not None:
            background_gdf.plot(ax=ax, color="white", edgecolor="black", linewidth=1.5, zorder=0)

        # Structure data list (adjusted to handle zones)
        formatted_cluster_data = []

        if mode == 'hard':
            if self.labels is None: raise ValueError("Model not fitted yet.")
            for i in range(self.n):
                idx = np.where(self.labels == i)[0]
                formatted_cluster_data.append({
                    'parent_color': None,  # assigned later
                    'free_points': {'indices': idx, 'shares': np.ones(len(idx), dtype=float)},
                    'border_points': {'indices': np.array([], dtype=int), 'shares': np.array([], dtype=float)},
                    'zones': []
                })
        else:
            if self.clusters is None:
                raise ValueError("Model not fitted yet.")
            for cluster in self.clusters:
                # 1. Identify "Internal" (free) points of the parent cluster
                # In your fit_zones logic, these are the original IDs minus parent borders.
                zone_ids = [idx for z in cluster.zones for idx in z.ids]
                free_ids = np.array(zone_ids, dtype=int)

                # 2. Extract border info (remains untouched by zones)
                border_indices = []
                border_shares = []
                if cluster.floor.index != -1 and cluster.floor.percentage > 1e-9:
                    border_indices.append(cluster.floor.index)
                    border_shares.append(cluster.floor.percentage)
                if cluster.ceil.index != -1 and cluster.ceil.percentage > 1e-9:
                    border_indices.append(cluster.ceil.index)
                    border_shares.append(cluster.ceil.percentage)

                # 3. Handle specific zone-level points (subpopulation indices)
                zones_list = []
                # Only use individual zone details if fit_zones was actually called (len > 1)
                # and we want to draw sub-convex hulls.
                if len(cluster.zones) > 1 and show_zone_sub_hulls:
                    for z in cluster.zones:
                        z_indices = np.array(z.ids, dtype=int)
                        zones_list.append({
                            'indices': z_indices,
                            'shares': np.ones(len(z_indices), dtype=float),
                            'label': z.label
                        })

                formatted_cluster_data.append({
                    'parent_color': None,
                    'free_points': {'indices': free_ids, 'shares': np.ones(len(free_ids), dtype=float)},
                    'border_points': {'indices': np.array(border_indices, dtype=int),
                                      'shares': np.array(border_shares, dtype=float)},
                    'zones': zones_list
                })

        k = len(formatted_cluster_data)

        # 1. Calculate Weighted Centroids (used for color sorting AND plotting)
        centroids = np.full((k, 2), np.nan, float)

        for i, c_data in enumerate(formatted_cluster_data):
            all_indices = np.concatenate([c_data['free_points']['indices'], c_data['border_points']['indices']])
            if len(all_indices) == 0: continue

            all_shares = np.concatenate([c_data['free_points']['shares'], c_data['border_points']['shares']])

            pts = self.population.coords[all_indices]
            # Probabilities must be adjusted by sharing weights
            adjusted_probs = self.population.inclusions[all_indices] * all_shares
            s = float(adjusted_probs.sum())
            centroids[i] = (pts * adjusted_probs[:, None]).sum(axis=0) / s if s > 0 else pts.mean(axis=0)

        # 2. Spatial Color Assignment (Lexicographical sort)
        palette_seq = [
            "#59A14F", "#E15759", "#1F77B4", "#FDD835",
            "#9467BD", "#8C564B", "#17BECF", "#FF7F0E", "#56B4E9",
            "#EF5350", "#CC79A7", "#FBC02D", "#7F3C8D", "#11A579",
            "#EF5350", "#FFC107"
        ]

        # Sort spatially (Top-to-bottom, left-to-right)
        order = np.lexsort((centroids[:, 0], -centroids[:, 1]))
        for rank, idx in enumerate(order):
            if rank < len(palette_seq):
                formatted_cluster_data[idx]['parent_color'] = palette_seq[rank % len(palette_seq)]
            else:
                formatted_cluster_data[idx]['parent_color'] = "#808080"  # Fallback

        # 3. Draw Clusters and Zones
        for i, c_data in enumerate(formatted_cluster_data):
            c = c_data['parent_color']
            if c is None: continue  # Skip empty/invalid clusters

            # --- Draw Internal Points and Soft Hull ---
            # Use all relevant indices for the main hull
            all_indices = np.concatenate([c_data['free_points']['indices'], c_data['border_points']['indices']])
            coords = self.population.coords[all_indices]
            all_shares = np.concatenate([c_data['free_points']['shares'], c_data['border_points']['shares']])
            probs = self.population.inclusions[all_indices] * all_shares

            # The overall structure (Soft border shape) is drawn first (Lowest Z-order)
            _draw_hull(coords, color=c, alpha=0.15, edge_color="black", lw=1.0)

            # --- Draw Internal Zones ---
            if c_data['zones']:
                # Plot internal sub-hulls to visualize fit_zones segmentation
                for zone in c_data['zones']:
                    if zone['indices'].size > 0:
                        zone_coords = self.population.coords[zone['indices']]
                        # Sub-hulls use slightly more opacity to differentiate them
                        _draw_hull(zone_coords, color=c, alpha=0.35, edge_color="none", lw=0.0)

                        if show_zone_labels:
                            _draw_zone_centroid_label(zone_coords, zone['label'])

            # --- Draw Points ---
            sizes = probs * size_scale
            if mode == 'soft' and len(c_data['border_points']['indices']) > 0:
                is_border = np.isin(all_indices, c_data['border_points']['indices'])
                # Free points of the parent cluster
                ax.scatter(coords[~is_border, 0], coords[~is_border, 1],
                           s=sizes[~is_border], color=c,
                           edgecolors="none", alpha=1.0, zorder=2)
                # Border points (Black) - Highest Z-order for points
                ax.scatter(coords[is_border, 0], coords[is_border, 1],
                           s=sizes[is_border], color="black",
                           edgecolors="none", alpha=1.0, zorder=3)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color=c,
                           edgecolors="none", alpha=1.0, zorder=2)

        # 4. Handle Centroids Visuals (drawn based on parent centroids)
        if connect_centroids and len(centroids) > 1:
            ax.plot(centroids[:, 0], centroids[:, 1],
                    color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=4)

        if show_centroids and len(centroids) > 0:
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       marker="^", color="black", s=80, zorder=5, label="Centroids")

        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"$X_1$")
        ax.set_ylabel(r"$X_2$")
        ax.set_title(f"FIP Balanced {self.n}-Means {mode.capitalize()} Clustering with Zones")
        return ax
