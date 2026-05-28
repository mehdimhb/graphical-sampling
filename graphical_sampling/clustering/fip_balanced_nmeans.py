from __future__ import annotations

import warnings
from typing import Literal
from itertools import pairwise
from scipy.special import logsumexp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
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



class _Clustering:
    def __init__(
        self,
        n_clusters: int,
        epsilon: float = 0.05,
        max_iter: int = 50,
        sinkhorn_iter: int =50,
        tol: float = 1e-4,
        n_init: int = 5,
        random_state: int | None = None,
    ):
        self.K = n_clusters
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sinkhorn_iter = sinkhorn_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.membership: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.objective: float | None = None

    @staticmethod
    def _compute_distances(coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return cdist(coords, centroids, metric='sqeuclidean')

    def _weighted_kmeans_pp(
            self, coords: np.ndarray, inclusions: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        N, D = coords.shape
        K = self.K

        centers = np.empty((K, D))

        idx = rng.integers(N)
        centers[0] = coords[idx]

        closest_dist = np.sum((coords - centers[0]) ** 2, axis=1)

        for k in range(1, K):
            probs = inclusions * closest_dist
            probs /= probs.sum()

            idx = rng.choice(N, p=probs)
            centers[k] = coords[idx]

            new_dist = np.sum((coords - centers[k]) ** 2, axis=1)
            closest_dist = np.minimum(closest_dist, new_dist)

        return centers

    def _sinkhorn_log(self, C, pi, b, f, g):

        eps = self.epsilon

        for _ in range(self.sinkhorn_iter):

            tmp = (g[None, :] - C) / eps
            f = eps * (np.log(pi) - logsumexp(tmp, axis=1))

            tmp = (f[:, None] - C) / eps
            g = eps * (np.log(b) - logsumexp(tmp, axis=0))

        return f, g

    @staticmethod
    def _hard_objective(coords: np.ndarray, centroids: np.ndarray, labels: np.ndarray, inclusion: np.ndarray) -> float:
        d = np.sum((coords - centroids[labels]) ** 2, axis=1)
        return np.sum(inclusion * d).item()

    def _fit_single(self, coords: np.ndarray, inclusion: np.ndarray, rng: np.random.Generator,
                    init_centroids: np.ndarray | None = None):
        N, D = coords.shape
        K = self.K
        b = np.ones(K)

        if init_centroids is not None:
            assert init_centroids.shape == (K, D), "init_centroids shape must be (n_clusters, n_features)"
            centroids = init_centroids.copy()
        else:
            centroids = self._weighted_kmeans_pp(coords, inclusion, rng)

        f = np.zeros(N)
        g = np.zeros(K)

        for _ in range(self.max_iter):
            C = self._compute_distances(coords, centroids)
            C = C / (np.median(C) + 1e-12)

            f, g = self._sinkhorn_log(C, inclusion, b, f, g)
            P = np.exp((f[:, None] + g[None, :] - C) / self.epsilon)

            new_centroids = P.T @ coords
            shift = np.linalg.norm(new_centroids - centroids)

            centroids = new_centroids
            if shift < self.tol:
                break

        R = P / inclusion[:, None]
        labels = np.argmax(R, axis=1)
        obj = self._hard_objective(coords, centroids, labels, inclusion)

        return centroids, R, labels, obj

    def fit(self, coords: np.ndarray, inclusions: np.ndarray,
            init_centroids: np.ndarray | None = None) -> _Clustering:
        rng = np.random.default_rng(self.random_state)

        best_obj = np.inf
        best_state = None

        n_runs = 1 if init_centroids is not None else self.n_init
        for _ in range(n_runs):
            centroids, R, labels, obj = self._fit_single(coords, inclusions, rng, init_centroids=init_centroids)
            if obj < best_obj:
                best_obj = obj
                best_state = (centroids, R, labels)

        self.centroids, self.membership, self.labels = best_state
        self.objective = best_obj
        return self
        
class FIPBalancedNMeans:
    def __init__(self, n: int, r_sample_per_cluster: int = 1, centroid_grid_x: int = None, n_init=50, tol=1e-9, max_iter=100, split_size=0.001,
                init_clust_method: Literal['weighted', 'expanded', 'ot'] = 'expanded',
                ) -> None:
        self.n = n
        assert n % r_sample_per_cluster == 0, "n must be divisible by r_sample_per_cluster"
        self.r = r_sample_per_cluster
        self.K = int(n // r_sample_per_cluster)
        self.centroid_grid_x = centroid_grid_x 
        
        self.pop: Population | None = None
        self.labels: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.path_order: np.ndarray | None = None
        self.membership: np.ndarray | None = None
        self.clusters: list[Cluster] = []
        self.tsp_solver = OpenTSPSolver()

        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter
        self.init_clust_method = init_clust_method
        self.split_size = split_size

    def fit(self, population: Population, init_centroids: np.ndarray | None = None) -> None:
        self.pop = population
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
            else np.zeros(coords.shape[1]) for i in range(self.K)
        ])

    def _get_labels_centroids_ot_kmeans(
            self,
            coords,
            inclusions,
            init_centroids=None
    ):
    
        clustering = _Clustering(
            n_clusters=self.K,
            epsilon=0.01,
            max_iter=100,
            sinkhorn_iter=100,
            tol=1e-4,
            n_init=5,
        )
    
        clustering.fit(coords, inclusions, init_centroids)
    
        return clustering.labels, clustering.centroids

    def fit_zones(
            self, 
            num_zones: int | tuple[int, int], 
            mode: Literal['cluster', 'sweep_xy', 'sweep_yx'],
            centroid_grid_x_zone: int | None = None  # <--- 1. NEW PARAMETER HERE
    ) -> None:
        if self.clusters is None:
            raise ValueError("The main clusters have not been fitted yet. Call .fit() first.")

        for i, cluster in enumerate(self.clusters):
            # --- 1. GATHER ALL INDICES AND FRACTIONAL SHARES ---
            c_indices = list(cluster.zones[0].indices)
            c_shares = list(cluster.zones[0].shares)

            if cluster.floor.index != -1 and cluster.floor.percentage > 1e-9:
                c_indices.append(cluster.floor.index)
                c_shares.append(cluster.floor.percentage)

            if cluster.ceil.index != -1 and cluster.ceil.percentage > 1e-9:
                c_indices.append(cluster.ceil.index)
                c_shares.append(cluster.ceil.percentage)

            c_indices_arr = np.array(c_indices, dtype=np.int64)
            c_shares_arr = np.array(c_shares, dtype=np.float64)

            sp = self.pop.subset(c_indices_arr, share=c_shares_arr)

            if isinstance(num_zones, tuple) and mode == 'cluster':
                raise ValueError(f"num_partitions must be a 'int' in the mode of cluster, not {type(num_zones)}")
            elif isinstance(num_zones, int) and mode != 'cluster':
                num_zones = (num_zones, num_zones)

            if mode == 'cluster':
                # <--- 2. PASS PARAMETER DOWN TO RECURSIVE FUNCTION HERE --->
                self.clusters[i].zones = self._get_zones_from_fbn(sp, c_shares_arr, num_zones, centroid_grid_x_zone)
            elif mode == 'sweep_xy':
                self.clusters[i].zones = self._get_zones_from_sweeping(sp, c_shares_arr, num_zones, x_first=True)
            elif mode == 'sweep_yx':
                self.clusters[i].zones = self._get_zones_from_sweeping(sp, c_shares_arr, num_zones, x_first=False)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Must be 'cluster' or 'sweep'.")

            # --- FIX: CLEAR THE BORDERS SO THEY ARE NOT DOUBLE COUNTED ---
            self.clusters[i].floor = Floor(index=-1, percentage=0.0)
            self.clusters[i].ceil = Ceil(index=-1, percentage=0.0)

    # def _get_zones_from_fbn(
    #         self, 
    #         subpopulation: Population, 
    #         c_shares: np.ndarray, 
    #         num_zones: int,
    #         centroid_grid_x_zone: int | None = None  # <--- 3. ADD TO SIGNATURE HERE
    # ) -> list[Zone]:

    #     zones_fbn = FIPBalancedNMeans(
    #         n=num_zones,
    #         centroid_grid_x=centroid_grid_x_zone,  # <--- 4. INJECT INTO NEW INSTANCE HERE
    #         n_init=self.n_init,
    #         tol=self.tol,
    #         max_iter=self.max_iter
    #     )
    #     zones_fbn.r = np.sum(subpopulation.inclusions) / num_zones
    #     zones_fbn.fit(subpopulation)

    #     zones = []
    #     for zone in zones_fbn.clusters:
    #         sc_indices = []
    #         sc_shares = []
            

    #         if zone.floor.index != -1:
    #             sc_indices.append(zone.floor.index)
    #             sc_shares.append(zone.floor.percentage)

    #         for z in zone.zones:
    #             for idx, share in zip(z.indices, z.shares):
    #                 sc_indices.append(idx)
    #                 sc_shares.append(share)
    #         if zone.ceil.index != -1:
    #             sc_indices.append(zone.ceil.index)
    #             sc_shares.append(zone.ceil.percentage)

    #         zones.append(
    #             Zone(
    #                 _indices=np.array(sc_indices),
    #                 _shares=np.array(sc_shares),
    #                 sort=np.arange(len(sc_indices)).tolist()
    #             )
    #         )

    #     return zones

    @staticmethod
    def _get_zones_indices_share(
            num_zones: int,
            indices: np.ndarray,
            shares: np.ndarray,
            probs: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if num_zones == 1:
            return [(indices, shares)]
            
        target_mass = np.sum(probs) / num_zones
        N = len(indices)
        
        zones = []
        curr_idx = 0
        
        if N == 0:
            return []
            
        unit_rem_prob = probs[0]
        
        for i in range(num_zones):
            zone_indices = []
            zone_shares_local = []
            
            mass_needed = target_mass
            
            if i == num_zones - 1:
                # Last zone takes all remaining fragments (prevents float rounding leftovers)
                while curr_idx < N:
                    if unit_rem_prob > 1e-12:
                        zone_indices.append(indices[curr_idx])
                        # FIX: Hard clip to ensure share never exceeds 1.0 due to float drift
                        frac = np.clip(unit_rem_prob / probs[curr_idx], 0.0, 1.0)
                        zone_shares_local.append(frac * shares[curr_idx])
                    curr_idx += 1
                    if curr_idx < N:
                        unit_rem_prob = probs[curr_idx]
            else:
                while mass_needed > 1e-12 and curr_idx < N:
                    # FIX: slightly wider tolerance (1e-10) to catch float overshoots
                    if unit_rem_prob <= mass_needed + 1e-10: 
                        if unit_rem_prob > 1e-12:
                            zone_indices.append(indices[curr_idx])
                            frac = np.clip(unit_rem_prob / probs[curr_idx], 0.0, 1.0)
                            zone_shares_local.append(frac * shares[curr_idx])
                        
                        mass_needed -= unit_rem_prob
                        # FIX: Prevent negative mass_needed from breaking the next loop
                        if mass_needed < 0: 
                            mass_needed = 0.0 
                            
                        curr_idx += 1
                        if curr_idx < N:
                            unit_rem_prob = probs[curr_idx]
                        else:
                            unit_rem_prob = 0.0
                    else:
                        # Consume exactly mass_needed, unit still has leftovers
                        zone_indices.append(indices[curr_idx])
                        frac = np.clip(mass_needed / probs[curr_idx], 0.0, 1.0)
                        zone_shares_local.append(frac * shares[curr_idx])
                        
                        unit_rem_prob -= mass_needed
                        mass_needed = 0.0
                        
            zones.append((
                np.array(zone_indices, dtype=np.int64),
                np.array(zone_shares_local, dtype=np.float64)
            ))
            
        return zones

    def _get_zones_from_sweeping(
            self, sp: Population, c_shares: np.ndarray, num_zones: tuple[int, int], x_first: bool
    ) -> list[Zone]:
        # THE FIX: Fetch raw un-multiplied inclusions from parent population
        raw_probs = self.pop.inclusions[sp.indices]

        sort = np.argsort(sp.coords[:, 0]) if x_first else np.argsort(sp.coords[:, 1])

        initial_zones = self._get_zones_indices_share(
            num_zones=num_zones[0] if x_first else num_zones[1],
            indices=sort,
            shares=c_shares[sort],
            # Target probability is raw inclusion * absolute share (No double multiply!)
            probs=raw_probs[sort] * c_shares[sort],
        )

        final_zones = []
        for i, (zone_indices, zone_share) in enumerate(initial_zones):
            sort = np.argsort(sp.coords[zone_indices][:, 1]) if x_first else np.argsort(sp.coords[zone_indices][:, 0])
            
            secondary_zones = self._get_zones_indices_share(
                num_zones=num_zones[1] if x_first else num_zones[0],
                indices=sort,
                shares=zone_share[sort],
                # Target probability is raw inclusion * the newly calculated absolute share
                probs=raw_probs[zone_indices][sort] * zone_share[sort],
            )
            
            for j, (sec_zone_indices, sec_zone_share) in enumerate(secondary_zones):
                new_zone = Zone(
                    _indices=sp.indices[zone_indices][sec_zone_indices],
                    _shares=sec_zone_share,
                    sort=np.arange(len(sec_zone_share)).tolist(),
                )
                vx = float(i) if x_first else float(j)
                vy = float(j) if x_first else float(i)
                new_zone.virtual_centroid = np.array([vx, vy])
                final_zones.append(new_zone)
                
        return final_zones

    def _get_zones_from_fbn(
            self, subpopulation: Population, c_shares: np.ndarray, num_zones: int, centroid_grid_x_zone: int | None = None
    ) -> list[Zone]:

        zones_fbn = FIPBalancedNMeans(
            n=num_zones, centroid_grid_x=centroid_grid_x_zone, 
            n_init=self.n_init, tol=self.tol, max_iter=self.max_iter
        )
        zones_fbn.r = np.sum(subpopulation.inclusions) / num_zones
        zones_fbn.fit(subpopulation)

        # Map local subpopulation index to its global share
        g2share = {idx: share for idx, share in zip(subpopulation.indices, c_shares)}

        zones = []
        for zone in zones_fbn.clusters:
            sc_indices, sc_shares = [], []

            if zone.floor.index != -1:
                sc_indices.append(zone.floor.index)
                sc_shares.append(zone.floor.percentage * g2share[zone.floor.index]) # Multiply absolute

            for z in zone.zones:
                for idx, share in zip(z.indices, z.shares):
                    sc_indices.append(idx)
                    sc_shares.append(share * g2share[idx])

            if zone.ceil.index != -1:
                sc_indices.append(zone.ceil.index)
                sc_shares.append(zone.ceil.percentage * g2share[zone.ceil.index])

            zones.append(Zone(_indices=np.array(sc_indices), _shares=np.array(sc_shares), sort=np.arange(len(sc_indices)).tolist()))

        return zones
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

         # --- grid initialization for centroids ---
            if self.centroid_grid_x is None:
                centroid_grid_x = int(np.ceil(np.sqrt(self.K)))
            else:
                centroid_grid_x = self.centroid_grid_x
            grid_x = np.linspace(coords.min(), coords.max(), centroid_grid_x)
            grid_y = np.linspace(coords.min(), coords.max(), int(np.ceil(self.K/centroid_grid_x)))
            gx, gy = np.meshgrid(grid_x, grid_y)
            grid_centers = np.column_stack([gx.ravel(), gy.ravel()])
            init_centers = grid_centers[:self.K]
            # -----------------------------------------
            # --------------------------------------------------
            # Weighted KMeans
            # --------------------------------------------------
            if self.init_clust_method == "weighted":

                probs_normalized = probs / probs.sum() * len(coords)

                if init_centroids is not None:
                    k_means = KMeans(
                        n_clusters=self.K,
                        init=init_centroids,
                        tol=self.tol,
                        max_iter=self.max_iter,
                        n_init=1
                    )

                    labels = k_means.fit_predict(coords, sample_weight=probs_normalized)
                    centroids = k_means.cluster_centers_
                    return labels, centroids

                best_error = np.inf
                best_labels = None
                best_centroids = None

                for _ in range(self.n_init):

                    k_means = KMeans(
                        n_clusters=self.K,
                        n_init=1,
                        tol=self.tol,
                        max_iter=self.max_iter
                    )

                    raw_labels = k_means.fit_predict(coords, sample_weight=probs_normalized)
                    raw_centroids = k_means.cluster_centers_

                    sums = np.array([probs[raw_labels == i].sum() for i in range(self.K)])
                    mean_abs_error = np.abs(sums - self.r).sum()

                    if mean_abs_error < best_error:
                        best_error = mean_abs_error
                        best_labels = raw_labels
                        best_centroids = raw_centroids

                return best_labels, best_centroids

            
            
            # --------------------------------------------------
            # Probability expansion + constrained KMeans
            # --------------------------------------------------
            if self.init_clust_method == "expanded":
                N = coords.shape[0]
            
                counts = (probs / self.split_size).round().astype(int)
                counts[counts == 0] = 1
            
                expanded_coords = np.repeat(coords, counts, axis=0)
                expanded_idx = np.repeat(np.arange(N), counts)
            
                cluster_size = max(1, len(expanded_idx) // self.K)
            
                kmeans = KMeansConstrained(
                    n_clusters=self.K,
                    size_min=cluster_size,
                    size_max=cluster_size + 1 if self.K > 1 else cluster_size,
                    init=init_centers,
                    n_init=1,
                    random_state=42,
                    n_jobs=-1
                )
            
                extended_labels = kmeans.fit_predict(expanded_coords)
            
                membership_counts = np.zeros((N, self.K), dtype=int)
                np.add.at(membership_counts, (expanded_idx, extended_labels), 1)
            
                labels = np.argmax(membership_counts, axis=1)
            
                centroids = np.array([
                    coords[labels == i].mean(axis=0) if np.any(labels == i)
                    else np.nan * coords[:1].mean(axis=0)
                    for i in range(self.K)
                ])
            
                return labels, centroids
            
            # --------------------------------------------------
            # Sinkhorn + constrained KMeans
            # --------------------------------------------------
            elif self.init_clust_method == "ot":
                
                labels_init, centroids_init = self._get_labels_centroids_ot_kmeans(
                    coords,
                    probs,
                    init_centers
                )
            
                return labels_init, centroids_init




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
            if curr_idx.size == 0:
                continue

            key = np.zeros(len(curr_idx))

            # Distance to PREV cluster
            if i > 0:
                prev_idx = clusters_idx.get(order[i - 1], np.array([], dtype=int))
                if prev_idx.size > 0:
                    d_prev = cdist(coords[curr_idx], coords[prev_idx]).min(axis=1)
                    key += d_prev

            # Distance to NEXT cluster
            if i < self.K - 1:
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
        target_mass = self.r
        thresholds = np.arange(1, self.K) * target_mass

        # Find split points
        split_indices = np.array(np.searchsorted(cum_probs, thresholds, side='left'))




        self._before_indices = full_indices

        final_clusters = []
        start_idx = 0
        split_idx = -1
        prev_border_remainder = 0.0
        prev_border_idx = -1

        for i in range(self.K):
            # Determine boundaries and fractional border ownership
            if i < self.K - 1:
                split_idx = min(split_indices[i], len(cum_probs) - 1)
                # split_idx = split_indices[i]

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

            # Convert internal indices to pop indices
            free_ids = pop_indices[free_indices] if pop_indices is not None else free_indices
            floor_id = pop_indices[prev_border_idx] if (
                        pop_indices is not None and prev_border_idx != -1) else prev_border_idx
            ceil_id = pop_indices[curr_border_idx] if (
                        pop_indices is not None and curr_border_idx != -1) else curr_border_idx

            final_clusters.append(
                Cluster(
                    label=int(order[i]),
                    zones=[
                        Zone(_indices=free_ids, _shares=np.ones_like(free_ids), sort=np.arange(len(free_ids)).tolist())
                    ],
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
        membership = np.zeros((N, self.K), dtype=float)

        if pop_indices is not None:
            g2l = {global_idx: local_idx for local_idx, global_idx in enumerate(pop_indices)}

            for cluster in clusters:
                for zone in cluster.zones:
                    if len(zone.indices) > 0:
                        local_ids = [g2l[i] for i in zone.indices]
                        # Safely accumulate the exact fractional shares
                        np.add.at(membership[:, cluster.label], local_ids, zone.shares)
                
                # These will only add anything if fit_zones hasn't cleared them yet
                if cluster.floor.index != -1 and cluster.floor.percentage > 1e-9:
                    membership[g2l[cluster.floor.index], cluster.label] += cluster.floor.percentage
                if cluster.ceil.index != -1 and cluster.ceil.percentage > 1e-9:
                    membership[g2l[cluster.ceil.index], cluster.label] += cluster.ceil.percentage
        else:
            for cluster in clusters:
                for zone in cluster.zones:
                    if len(zone.indices) > 0:
                        # Safely accumulate the exact fractional shares
                        np.add.at(membership[:, cluster.label], zone.indices, zone.shares)
                
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
        size_scale: float = 700.0,
        figsize: tuple[int, int] = (8, 6),
        dpi: int = 150,
        show_zone_sub_hulls: bool = True,
        show_zone_labels: bool = True,
        zone_color_mode: bool = False,
        show_cluster_mass: bool = False,
        point_alpha: float = 0.78,
        hull_alpha: float = 0.22,
        zone_hull_alpha: float = 0.42,
        point_size_min: float = 8.0,
        point_size_max: float = 38.0,
        hull_edge_lw: float = 0.80,
        background_lw: float = 1.20
    ) -> plt.Axes:
        """
        Plot the FIP-balanced n-means clustering.

        Main visual controls
        --------------------
        size_scale:
            Controls point sizes when inclusion probabilities are unequal.
        point_alpha:
            Transparency of points.
        hull_alpha:
            Transparency of parent cluster polygons.
        zone_hull_alpha:
            Transparency of internal zone polygons.
        point_size_min, point_size_max:
            Lower and upper bounds for displayed point sizes.
        zone_color_mode:
            If True, zones are colored by their zone label.
            If False, each parent cluster has its own color.
        """

        def _draw_hull(
                points: np.ndarray,
                color: str,
                alpha: float,
                edge_color: str | None,
                lw: float,
                zorder: int
        ):
            if points.shape[0] < 3:
                return None
            try:
                hull = ConvexHull(points)
                vertices = points[hull.vertices]
            except (QhullError, ValueError):
                return None

            ax.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor=edge_color,
                    linewidth=lw,
                    zorder=zorder
                )
            )
            return vertices.mean(axis=0)

        def _draw_zone_centroid_label(points: np.ndarray, label: int):
            if points.shape[0] == 0:
                return
            centroid = points.mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(label + 1),
                fontsize=12,
                color="0.25",
                ha="center",
                va="center",
                zorder=10
            )

        def _bounded_sizes(raw_sizes: np.ndarray) -> np.ndarray:
            raw_sizes = np.asarray(raw_sizes, dtype=float)
            if raw_sizes.size == 0:
                return raw_sizes
            return np.clip(raw_sizes, point_size_min, point_size_max)

        def _generate_palette(n: int) -> list[str]:
            # Similar to your target figure: bright, clear, but still soft.
            base = [
                "#E41A1C",  # red
                "#377EB8",  # blue
                "#4DAF4A",  # green
                "#984EA3",  # purple
                "#FF7F00",  # orange
                "#FFD92F",  # yellow
                "#A65628",  # brown
                "#F781BF",  # pink
                "#00BFC4",  # cyan
                "#66C2A5",  # mint
                "#FC8D62",  # salmon
                "#8DA0CB",  # lavender-blue
                "#E78AC3",  # light magenta
                "#A6D854",  # light green
                "#E5C494",  # tan
                "#B3B3B3",  # gray
            ]
            return [base[i % len(base)] for i in range(n)]

        def _generate_zone_palette(n: int) -> list[str]:
            base = [
                # "#4DAF4A",  # green
                "#377EB8",  # blue
                "#E41A1C",  # red
                "#FFD92F",  # yellow
                "#984EA3",  # purple
                "#FF7F00",  # orange
            ]
            return [base[i % len(base)] for i in range(n)]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # ---------------------------------------------------------
        # Background boundary
        # ---------------------------------------------------------
        if background_gdf is not None:
            background_gdf.boundary.plot(
            ax=ax,
            color="0.25",
            linewidth=background_lw,
            zorder=0
        )

        # ---------------------------------------------------------
        # Build plot data
        # ---------------------------------------------------------
        formatted_cluster_data = []

        if mode == 'hard':
            if self.labels is None:
                raise ValueError("Model not fitted yet.")

            for i in range(self.K):
                idx = np.where(self.labels == i)[0]
                formatted_cluster_data.append({
                    'parent_color': None,
                    'free_points': {
                        'indices': idx,
                        'shares': np.ones(len(idx), dtype=float)
                    },
                    'border_points': {
                        'indices': np.array([], dtype=int),
                        'shares': np.array([], dtype=float)
                    },
                    'zones': []
                })

        else:
            if self.clusters is None:
                raise ValueError("Model not fitted yet.")

            for cluster in self.clusters:
                free_ids = []
                free_shares = []

                for z in cluster.zones:
                    free_ids.extend(z.indices)
                    free_shares.extend(z.shares)

                free_ids = np.array(free_ids, dtype=int)
                free_shares = np.array(free_shares, dtype=float)
                
                border_indices = []
                border_shares = []

                if cluster.floor.index != -1 and cluster.floor.percentage > 1e-9:
                    border_indices.append(cluster.floor.index)
                    border_shares.append(cluster.floor.percentage)

                if cluster.ceil.index != -1 and cluster.ceil.percentage > 1e-9:
                    border_indices.append(cluster.ceil.index)
                    border_shares.append(cluster.ceil.percentage)

                zones_list = []
                if len(cluster.zones) > 1 and show_zone_sub_hulls:
                    for zone_index, z in enumerate(cluster.zones):
                        z_indices = np.array(z.indices, dtype=int)
                        zones_list.append({
                            'indices': z_indices,
                            'shares': np.ones(len(z_indices), dtype=float),
                            'label': zone_index
                        })

                formatted_cluster_data.append({
                    'parent_color': None,
                    'free_points': {
                        'indices': free_ids,
                        'shares': free_shares
                    },
                    'border_points': {
                        'indices': np.array(border_indices, dtype=int),
                        'shares': np.array(border_shares, dtype=float)
                    },
                    'zones': zones_list
                })

        k = len(formatted_cluster_data)

        # ---------------------------------------------------------
        # Parent-cluster centroids for stable spatial color ordering
        # ---------------------------------------------------------
        centroids = np.full((k, 2), np.nan, dtype=float)

        for i, c_data in enumerate(formatted_cluster_data):
            all_indices = np.concatenate([
                c_data['free_points']['indices'],
                c_data['border_points']['indices']
            ])

            if len(all_indices) == 0:
                continue

            all_shares = np.concatenate([
                c_data['free_points']['shares'],
                c_data['border_points']['shares']
            ])

            pts = self.pop.coords[all_indices]
            adjusted_probs = self.pop.inclusions[all_indices] * all_shares
            total_prob = float(adjusted_probs.sum())

            if total_prob > 0:
                centroids[i] = (pts * adjusted_probs[:, None]).sum(axis=0) / total_prob
            else:
                centroids[i] = pts.mean(axis=0)

        parent_palette = _generate_palette(k)

        max_num_zones = max(
            len(c_data['zones']) if len(c_data['zones']) > 0 else 1
            for c_data in formatted_cluster_data
        )
        zone_palette = _generate_zone_palette(max_num_zones)

        valid = ~np.isnan(centroids[:, 0])
        order = np.lexsort((centroids[valid, 0], -centroids[valid, 1]))
        valid_indices = np.where(valid)[0][order]

        for rank, idx in enumerate(valid_indices):
            formatted_cluster_data[idx]['parent_color'] = parent_palette[rank]

        # ---------------------------------------------------------
        # Draw clusters, zones, and points
        # ---------------------------------------------------------
        for c_data in formatted_cluster_data:
            c = c_data['parent_color']
            if c is None:
                continue

            all_indices = np.concatenate([
                c_data['free_points']['indices'],
                c_data['border_points']['indices']
            ])

            if len(all_indices) == 0:
                continue

            all_shares = np.concatenate([
                c_data['free_points']['shares'],
                c_data['border_points']['shares']
            ])

            coords = self.pop.coords[all_indices]
            probs = self.pop.inclusions[all_indices] * all_shares

            # Parent hull
            _draw_hull(
                coords,
                color=c,
                alpha=0.10,
                edge_color="0.050",
                lw=hull_edge_lw,
                zorder=1
            )

            # Internal zone hulls
            if c_data['zones']:
                for zone in c_data['zones']:
                    if zone['indices'].size == 0:
                        continue

                    zone_indices = zone['indices']
                    zone_coords = self.pop.coords[zone_indices]

                    if zone_color_mode:
                        zone_color = zone_palette[zone['label'] % len(zone_palette)]
                    else:
                        zone_color = c

                    _draw_hull(
                        zone_coords,
                        color=zone_color,
                        alpha=.30,
                        edge_color=None,
                        lw=0.0,
                        zorder=2
                    )

                    if show_zone_labels:
                        _draw_zone_centroid_label(zone_coords, zone['label'])

            # Points
            if zone_color_mode and c_data['zones']:
                for zone in c_data['zones']:
                    if zone['indices'].size == 0:
                        continue

                    zone_indices = zone['indices']
                    zone_coords = self.pop.coords[zone_indices]
                    zone_color = zone_palette[zone['label'] % len(zone_palette)]

                    zone_sizes = _bounded_sizes(
                        self.pop.inclusions[zone_indices] * size_scale
                    )

                    ax.scatter(
                        zone_coords[:, 0],
                        zone_coords[:, 1],
                        s=zone_sizes,
                        color=zone_color,
                        edgecolors="none",
                        alpha=point_alpha,
                        zorder=4
                    )

            else:
                point_sizes = _bounded_sizes(probs * size_scale)

                if mode == 'soft' and len(c_data['border_points']['indices']) > 0:
                    is_border = np.isin(
                        all_indices,
                        c_data['border_points']['indices']
                    )

                    ax.scatter(
                        coords[~is_border, 0],
                        coords[~is_border, 1],
                        s=point_sizes[~is_border],
                        color=c,
                        edgecolors="none",
                        alpha=point_alpha,
                        zorder=4
                    )

                    ax.scatter(
                        coords[is_border, 0],
                        coords[is_border, 1],
                        s=point_sizes[is_border] * 1.15,
                        color="black",
                        edgecolors="none",
                        alpha=0.85,
                        zorder=5
                    )

                else:
                    ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        s=point_sizes,
                        color=c,
                        edgecolors="none",
                        alpha=point_alpha,
                        zorder=4
                    )

        # ---------------------------------------------------------
        # Cluster labels: total inclusion probability in each cluster
        # ---------------------------------------------------------
        if show_cluster_mass:
            for c_data in formatted_cluster_data:
                all_indices = np.concatenate([
                    c_data['free_points']['indices'],
                    c_data['border_points']['indices']
                ])

                all_shares = np.concatenate([
                    c_data['free_points']['shares'],
                    c_data['border_points']['shares']
                ])

                if len(all_indices) == 0:
                    continue

                pts = self.pop.coords[all_indices]
                probs = self.pop.inclusions[all_indices] * all_shares
                total_pi = probs.sum()

                center = (pts * probs[:, None]).sum(axis=0) / total_pi

                ax.text(
                    center[0],
                    center[1],
                    rf"{total_pi:.1f}",
                    fontsize=7,
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.20",
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.45,
                        linewidth=0.5
                    ),
                    zorder=20
                )

        # ---------------------------------------------------------
        # Optional centroids and centroid path
        # ---------------------------------------------------------
        if connect_centroids and np.sum(valid) > 1:
            ordered_centroids = centroids[valid_indices]
            ax.plot(
                ordered_centroids[:, 0],
                ordered_centroids[:, 1],
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.65,
                zorder=6
            )

        if show_centroids and np.sum(valid) > 0:
            ax.scatter(
                centroids[valid, 0],
                centroids[valid, 1],
                marker="^",
                color="black",
                s=65,
                alpha=0.90,
                zorder=7,
                label="Centroids"
            )

        # ---------------------------------------------------------
        # Clean academic styling
        # ---------------------------------------------------------
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(r"$X_1$")
        ax.set_ylabel(r"$X_2$")

        ax.set_title(
            f"FIP-balanced {self.n}-means clustering",
            fontsize=11
        )

        return ax