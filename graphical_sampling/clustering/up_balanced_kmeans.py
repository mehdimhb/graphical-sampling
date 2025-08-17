import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.neighbors import NearestNeighbors
from k_means_constrained import KMeansConstrained

# project-local imports (adjust paths to your project layout)
from ..population import Population
from .shortest_path import shortest_through_all_points


class UPBalancedKMeans:
    """
    UP-Balanced K-Means with natural inter-cluster connections.

    Pipeline
    --------
    1) Expand each point proportional to its probability.
    2) Constrained K-Means on the expanded set.
    3) Map assignments back → soft membership of original points.
    4) Order clusters along a path (shortest_through_all_points).
    5) Within each cluster block, sort by 1-NN distance to previous/next clusters,
       and *pin endpoints* so joins are natural.
    6) Split along cumulative-probability thresholds to balance mass.
       - Fractions are snapped to {0,1}.
       - If a border's share equals 1, the point is promoted to FREE (no border).
    7) Emit hard labels and final centroids.

    Parameters
    ----------
    k : int
        Number of clusters.
    split_size : float
        Smaller → more expansion weight.
    nn_algo : {"auto","kd_tree","ball_tree"}
        Algorithm for NearestNeighbors (1-NN) lookups.
    nn_leaf_size : int
        Leaf size for the neighbor tree.
    snap_atol, snap_rtol : float
        Tolerances for snapping border fractions to {0, 1}.
    """

    def __init__(self, k: int, split_size: float = 0.001,
                 nn_algo: str = "auto", nn_leaf_size: int = 40,
                 snap_atol: float = 1e-12, snap_rtol: float = 1e-9):
        assert k >= 1, "k must be >= 1"
        self.k: int = k
        self.split_size: float = split_size

        # results
        self.N: Optional[int] = None
        self.membership: Optional[np.ndarray] = None   # (N, k)
        self.labels: Optional[np.ndarray] = None       # (N,)
        self.centroids: Optional[np.ndarray] = None    # (k, d)
        self.clusters: Optional[Dict[int, dict]] = None

        # NN + snapping controls
        self.nn_algo = nn_algo
        self.nn_leaf_size = nn_leaf_size
        self.snap_atol = snap_atol
        self.snap_rtol = snap_rtol

    # ---------- helpers -----------------------------------------------------

    def _generate_expanded_coords(
        self, coords: np.ndarray, probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Repeat each point proportional to its probability."""
        counts: np.ndarray = (probs / self.split_size).round().astype(int)
        counts[counts == 0] = 1
        expanded_coords: np.ndarray = np.repeat(coords, counts, axis=0)
        expanded_idx: np.ndarray = np.repeat(np.arange(self.N), counts)
        return expanded_coords, expanded_idx, counts

    def _generate_membership(
        self, extended_labels: np.ndarray, expanded_idx: np.ndarray,
        counts: np.ndarray
    ) -> np.ndarray:
        """Soft membership of original points to clusters."""
        membership_counts = np.zeros((self.N, self.k), dtype=int)
        np.add.at(membership_counts, (expanded_idx, extended_labels), 1)
        membership = membership_counts / counts[:, None]
        return membership

    def _nearest_dists_to_set(
        self, coords: np.ndarray, query_idx: np.ndarray, ref_idx: np.ndarray
    ) -> np.ndarray:
        """Return 1-NN Euclidean distance from each query point to ref set."""
        if query_idx.size == 0:
            return np.zeros(0, dtype=float)
        if ref_idx.size == 0:
            return np.zeros(query_idx.shape[0], dtype=float)

        X = coords[query_idx]
        Y = coords[ref_idx]

        algo = self.nn_algo
        if algo == "auto":
            # KD-tree works well in low/moderate dims; else Ball-tree
            algo = "kd_tree" if coords.shape[1] <= 10 else "ball_tree"

        nbrs = NearestNeighbors(
            n_neighbors=1,
            algorithm=algo,
            leaf_size=self.nn_leaf_size,
            metric="minkowski", p=2
        ).fit(Y)
        dists, _ = nbrs.kneighbors(X, return_distance=True)
        return dists.ravel()

    def _snap01(self, x: np.ndarray) -> np.ndarray:
        """Clamp to [0,1] and snap values very close to 0 or 1 exactly."""
        if x.size == 0:
            return x
        y = np.clip(x, 0.0, 1.0).astype(float, copy=True)
        near0 = np.isclose(y, 0.0, atol=self.snap_atol, rtol=self.snap_rtol)
        near1 = np.isclose(y, 1.0, atol=self.snap_atol, rtol=self.snap_rtol)
        y[near0] = 0.0
        y[near1] = 1.0
        return y

    def _generate_clusters(
        self,
        order: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        probs: np.ndarray,
        pop_indices: Optional[np.ndarray] = None,
    ) -> Dict[int, dict]:
        """
        Build a 'mega order' by sorting each cluster's points using
        nearest-neighbor distances to adjacent clusters. Pin endpoints
        to guarantee natural joins. Then apply the cumsum-based quota
        split and return per-cluster assignments + border splits.

        Promotion rule:
          - If a border share equals 1.0, that index is promoted to FREE
            for the cluster and the border entry is removed.
          - If a border share equals 0.0, the border entry is removed.
        """
        N = len(probs)
        eps = 1e-12

        # indices per label along the path
        label_to_indices = {int(lab): np.flatnonzero(labels == lab) for lab in order}

        mega_blocks = []
        for i, lab in enumerate(order):
            lab = int(lab)
            idx = label_to_indices.get(lab, np.array([], dtype=int))
            if idx.size == 0:
                mega_blocks.append(idx)
                continue

            has_prev = (i > 0) and (label_to_indices.get(int(order[i-1]), np.array([], int)).size > 0)
            has_next = (i < self.k - 1) and (label_to_indices.get(int(order[i+1]), np.array([], int)).size > 0)

            # 1-NN distances to adjacent clusters
            if has_prev:
                prev_lab = int(order[i - 1])
                d_prev_min = self._nearest_dists_to_set(coords, idx, label_to_indices[prev_lab])
            else:
                d_prev_min = None

            if has_next:
                next_lab = int(order[i + 1])
                d_next_min = self._nearest_dists_to_set(coords, idx, label_to_indices[next_lab])
            else:
                d_next_min = None

            # sorting key within the cluster block
            if has_prev and has_next:
                key = d_prev_min - d_next_min
            elif has_next:
                key = -d_next_min
            elif has_prev:
                key = d_prev_min
            else:
                key = np.zeros(idx.shape[0])

            block = idx[np.argsort(key, kind="stable")]

            # pin endpoints to enforce natural joins
            if has_prev:
                first_idx = idx[np.argmin(d_prev_min)]
                block = np.concatenate(([first_idx], block[block != first_idx]))
            if has_next:
                last_idx = idx[np.argmin(d_next_min)]
                mask = (block != last_idx)
                block = np.concatenate((block[mask], [last_idx]))

            mega_blocks.append(block)

        mega_order = np.concatenate(mega_blocks) if mega_blocks else np.array([], dtype=int)
        mega_actual_indices = pop_indices[mega_order] if pop_indices is not None else mega_order

        # ---- cumsum quota split ---------------------------------------------------
        if N == 0:
            return {int(lab): {'free': np.array([], dtype=int),
                               'border': {'floor_index': -1, 'floor_percentage': 0.0,
                                          'ceil_index': -1, 'ceil_percentage': 0.0}}
                    for lab in order}

        p_sorted = probs[mega_order]
        cs = np.cumsum(p_sorted)

        probs_total = round(float(np.sum(probs)))
        m = round(probs_total / self.k, 9) if self.k > 0 else 0.0

        thresholds = np.arange(m, probs_total + m, step=m, dtype=float)  # m, 2m, ..., ~probs_total
        border_pos = np.searchsorted(cs, thresholds, side='left')        # positions in mega_order

        # split each interior border i=1..k-1 into ceil(fills left) / floor(remainder to right)
        pos_1_to_km1 = border_pos[:-1]
        mass_before = np.where(pos_1_to_km1 > 0, cs[pos_1_to_km1 - 1], 0.0)
        border_idx_at_pos = mega_order[pos_1_to_km1] if pos_1_to_km1.size else np.array([], dtype=int)
        border_prob = probs[border_idx_at_pos] if pos_1_to_km1.size else np.array([], dtype=float)
        need = thresholds[:-1] - mass_before

        with np.errstate(divide='ignore', invalid='ignore'):
            ceil_raw = np.clip(need / np.maximum(border_prob, eps), 0.0, 1.0)

        # snap tiny epsilons to exact 0 or 1
        ceil_pct_for_prev = self._snap01(ceil_raw)
        floor_pct_for_curr = 1.0 - ceil_pct_for_prev

        # assemble outputs per cluster
        clusters: Dict[int, dict] = {}
        for c, lab in enumerate(order):
            lab = int(lab)

            # Defaults
            floor_index = -1
            floor_percentage = 0.0
            ceil_index = -1
            ceil_percentage = 0.0

            # Determine free slice limits; we may widen them based on promotion rule
            if c == 0:
                start_pos = 0
            else:
                pos_in = int(border_pos[c - 1])
                fp = float(floor_pct_for_curr[c - 1])

                if fp == 0.0:
                    # nothing comes in from the left at this border
                    start_pos = pos_in + 1
                elif fp == 1.0:
                    # full ownership → promote to FREE by including the border index
                    start_pos = pos_in      # include border index in free
                else:
                    # fractional share → keep as border
                    floor_index = int(mega_actual_indices[pos_in])
                    floor_percentage = fp
                    start_pos = pos_in + 1  # free starts after incoming border

            if c == self.k - 1:
                end_pos = N
            else:
                pos_out = int(border_pos[c])
                cp = float(ceil_pct_for_prev[c])

                if cp == 0.0:
                    # nothing goes out to the right at this border
                    end_pos = pos_out
                elif cp == 1.0:
                    # full ownership → promote to FREE by extending free to include border index
                    end_pos = pos_out + 1
                else:
                    # fractional share → keep as border
                    ceil_index = int(mega_actual_indices[pos_out])
                    ceil_percentage = cp
                    end_pos = pos_out

            # indices strictly between (or widened to include promoted borders)
            free_list = mega_actual_indices[start_pos:end_pos].tolist() if end_pos > start_pos else []

            clusters[lab] = {
                'free': np.array(free_list, dtype=int),
                'border': {
                    'floor_index': floor_index,
                    'floor_percentage': floor_percentage,
                    'ceil_index': ceil_index,
                    'ceil_percentage': ceil_percentage,
                }
            }

        return clusters

    def _generate_labels(self, clusters: Dict[int, dict], probs: np.ndarray) -> np.ndarray:
        """
        Build hard labels from cluster splits by turning each split into
        fractional memberships and taking argmax.
        """
        N = probs.shape[0]
        k = self.k
        frac = np.zeros((N, k), dtype=float)

        for lab, info in clusters.items():
            lab = int(lab)
            free = info['free']
            if free.size:
                frac[free, lab] += 1.0

            b = info['border']
            fi, fp = int(b['floor_index']), float(b['floor_percentage'])
            ci, cp = int(b['ceil_index']), float(b['ceil_percentage'])

            if fi != -1 and fp > 0.0:
                frac[fi, lab] += fp
            if ci != -1 and cp > 0.0:
                frac[ci, lab] += cp

        labels = np.full(N, -1, dtype=int)
        best_lab = np.argmax(frac, axis=1)
        has_any = (frac.max(axis=1) > 0.0)
        labels[has_any] = best_lab[has_any]
        return labels

    # ---------- public API --------------------------------------------------

    def fit(self, population: Population) -> None:
        """Fit the model to a Population (expects .coords, .probs, and .N)."""
        coords: np.ndarray = population.coords
        probs: np.ndarray = population.probs
        self.N = population.N

        # (1) expand by probabilities
        expanded_coords, expanded_idx, counts = self._generate_expanded_coords(coords, probs)

        # (2) constrained K-Means on expanded set
        cluster_size: int = max(1, len(expanded_idx) // self.k)
        kmeans = KMeansConstrained(
            n_clusters=self.k,
            size_min=cluster_size,
            size_max=cluster_size + 1 if self.k > 1 else cluster_size,
            n_jobs=-1,
            random_state=42
        )
        extended_labels: np.ndarray = kmeans.fit_predict(expanded_coords)

        # (3) soft membership back to original
        self.membership = self._generate_membership(extended_labels, expanded_idx, counts)

        # (4) raw hard labels + raw centroids
        raw_labels: np.ndarray = np.argmax(self.membership, axis=1)
        raw_centroids: np.ndarray = np.array([
            coords[raw_labels == i].mean(axis=0) if np.any(raw_labels == i)
            else np.nan * coords[:1].mean(axis=0)
            for i in range(self.k)
        ])

        # (5) order clusters along a path
        ordered_labels = shortest_through_all_points(raw_centroids)

        # (6) build clusters with NN-based joins + quota split (+ promotion rule)
        clusters = self._generate_clusters(ordered_labels, raw_labels, coords, probs, population.indices)

        # (7) final hard labels and centroids
        if population.indices is None:
            self.labels = self._generate_labels(clusters, probs)
            self.centroids = np.array([
                coords[self.labels == i].mean(axis=0) if np.any(self.labels == i)
                else np.nan * coords[:1].mean(axis=0)
                for i in range(self.k)
            ])

        self.clusters = list(clusters.values())
