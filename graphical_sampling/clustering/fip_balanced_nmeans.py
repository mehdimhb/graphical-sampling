import warnings
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, BisectingKMeans
from numba import jit

from ..population import Population


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
        self.clusters: list[dict] | None = None
        self.tsp_solver = OpenTSPSolver()

        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, population: Population, init_centroids: np.ndarray | None = None) -> None:
        self.population = population
        coords = population.coords
        probs = population.probs
        N = len(coords)

        # 1. Initial Clustering
        raw_labels, raw_centroids = self._get_labels_centroids(coords, probs, init_centroids)

        # 2. TSP Ordering
        # returns a linear path [Start -> ... -> End]
        self.path_order = self.tsp_solver.solve(raw_centroids)

        # 3. Exact Balanced Clusters
        # Sort internally based on the linear path neighbors & split by cumulative probability
        self.clusters = self._generate_exact_clusters(
            self.path_order, raw_labels, coords, probs, population.indices
        )

        # 4. Finalize Outputs
        self.membership = self._generate_membership(self.clusters, N)
        self.labels = np.argmax(self.membership, axis=1)

        # Recalculate centroids based on final hard labels
        self.centroids = np.array([
            coords[self.labels == i].mean(axis=0) if np.any(self.labels == i)
            else np.zeros(coords.shape[1]) for i in range(self.n)
        ])

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
    ) -> list[dict]:

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

        if not mega_indices: return []
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

            final_clusters.append({
                'label': int(order[i]),
                'free': free_ids,
                'border': {
                    'floor_index': int(floor_id),
                    'floor_percentage': float(prev_border_remainder),
                    'ceil_index': int(ceil_id),
                    'ceil_percentage': float(frac_curr)
                }
            })

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

    def _generate_membership(self, clusters, N) -> np.ndarray:
        membership = np.zeros((N, self.n), dtype=float)
        for i, c in enumerate(clusters):
            # Free
            if len(c['free']) > 0:
                membership[c['free'], i] = 1.0
            # Border
            b = c['border']
            if b['floor_index'] != -1 and b['floor_percentage'] > 1e-9:
                membership[b['floor_index'], i] += b['floor_percentage']
            if b['ceil_index'] != -1 and b['ceil_percentage'] > 1e-9:
                membership[b['ceil_index'], i] += b['ceil_percentage']
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
            dpi: int = 100
    ) -> plt.Axes:
        """
        Plot the clustering result.

        Args:
            show_centroids: If True, plots centroids as black triangles.
            connect_centroids: If True, connects centroids with a dashed line
                               (Order: TSP path in Soft mode; Label ID in Hard mode).
        """

        def _draw_hull(points: np.ndarray, color: str, alpha: float, edge_color: str, lw: float):
            if points.shape[0] < 3:
                return None
            try:
                hull = ConvexHull(points)
                verts = points[hull.vertices]
            except (QhullError, ValueError):
                return None
            ax.add_patch(Polygon(verts, closed=True, facecolor=color, alpha=alpha,
                                edgecolor=edge_color, lw=lw))
            return verts.mean(axis=0)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if background_gdf is not None:
            background_gdf.plot(ax=ax, color="white", edgecolor="black", linewidth=1.5, zorder=0)

        # Build cluster data list
        cluster_data = []

        if mode == 'hard':
            if self.labels is None: raise ValueError("Model not fitted yet.")
            for i in range(self.n):
                idx = np.where(self.labels == i)[0]
                cluster_data.append({
                    'indices': idx,
                    'shares': np.ones(len(idx), dtype=float),
                    'border_indices': np.array([], dtype=int)
                })
        else:
            if self.clusters is None: raise ValueError("Model not fitted yet.")
            for c_info in self.clusters:
                indices, shares, border_indices = [], [], []

                if len(c_info['free']) > 0:
                    indices.extend(c_info['free'])
                    shares.extend([1.0] * len(c_info['free']))

                b = c_info['border']
                if b['floor_index'] != -1 and b['floor_percentage'] > 1e-9:
                    indices.append(b['floor_index'])
                    shares.append(b['floor_percentage'])
                    border_indices.append(b['floor_index'])

                if b['ceil_index'] != -1 and b['ceil_percentage'] > 1e-9:
                    indices.append(b['ceil_index'])
                    shares.append(b['ceil_percentage'])
                    border_indices.append(b['ceil_index'])

                cluster_data.append({
                    'indices': np.array(indices, dtype=int),
                    'shares': np.array(shares, dtype=float),
                    'border_indices': np.array(border_indices, dtype=int)
                })

        k = len(cluster_data)

        # 1. Calculate Weighted Centroids (used for color sorting AND plotting)
        centroids = np.full((k, 2), np.nan, float)

        for i, c_data in enumerate(cluster_data):
            if len(c_data['indices']) == 0: continue
            pts = self.population.coords[c_data['indices']]
            w   = self.population.probs[c_data['indices']] * c_data['shares']
            s   = float(w.sum())
            centroids[i] = (pts * w[:, None]).sum(axis=0) / s if s > 0 else pts.mean(axis=0)

        valid = ~np.isnan(centroids).any(axis=1)

        # 2. Color Palette & Assignment
        palette4 = ["#59A14F", "#E15759", "#1F77B4", "#FDD835"]
        palette_seq = [
            "#59A14F", "#E15759", "#1F77B4", "#FDD835",
            "#9467BD", "#8C564B", "#17BECF", "#FF7F0E", "#56B4E9",
            "#EF5350", "#CC79A7", "#FBC02D", "#7F3C8D", "#11A579",
            "#EF5350", "#FFC107"
        ]
        colors_for_cluster = ["#808080"] * k

        if k == 4 and valid.all():
            y_mid = np.median(centroids[:, 1])
            top_idx = np.where(centroids[:, 1] >= y_mid)[0]
            bot_idx = np.where(centroids[:, 1] <  y_mid)[0]

            if top_idx.size != 2 or bot_idx.size != 2:
                order_y = np.argsort(-centroids[:, 1])
                top_idx = order_y[:2]; bot_idx = order_y[2:]

            tl = top_idx[np.argmin(centroids[top_idx, 0])]
            tr = top_idx[np.argmax(centroids[top_idx, 0])]
            bl = bot_idx[np.argmin(centroids[bot_idx, 0])]
            br = bot_idx[np.argmax(centroids[bot_idx, 0])]

            for idx, col in zip([tl, tr, bl, br], palette4):
                colors_for_cluster[idx] = col
        else:
            if valid.any():
                # Spatial sort for colors
                order = np.lexsort((centroids[:, 0], -centroids[:, 1]))
            else:
                order = np.arange(k)
            for rank, idx in enumerate(order):
                colors_for_cluster[idx] = palette_seq[rank % len(palette_seq)]

        # 3. Draw Clusters
        for i, c_data in enumerate(cluster_data):
            if len(c_data['indices']) == 0: continue

            color = colors_for_cluster[i]
            coords = self.population.coords[c_data['indices']]
            probs  = self.population.probs[c_data['indices']] * c_data['shares']

            _draw_hull(coords, color=color, alpha=0.20, edge_color="black", lw=1.0)

            sizes = probs * size_scale

            if mode == 'soft' and len(c_data['border_indices']) > 0:
                is_border = np.isin(c_data['indices'], c_data['border_indices'])
                # Standard points
                ax.scatter(coords[~is_border, 0], coords[~is_border, 1],
                           s=sizes[~is_border], color=color,
                           edgecolors="none", alpha=1.0, zorder=2)
                # Border points (Black)
                ax.scatter(coords[is_border, 0], coords[is_border, 1],
                           s=sizes[is_border], color="black",
                           edgecolors="none", alpha=1.0, zorder=3)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color=color,
                           edgecolors="none", alpha=1.0, zorder=2)

        # 4. Handle Centroids Visuals
        if connect_centroids and len(centroids) > 1:
            # Connect in the order they appear in the list (TSP path order for Soft)
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
        ax.set_title(f"FIP Balanced {self.n}-Means {mode.capitalize()} Clustering")
        return ax
