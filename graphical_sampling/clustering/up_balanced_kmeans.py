import numpy as np
from k_means_constrained import KMeansConstrained
from typing import Optional, List, Tuple
from ..population import Population
from .shortest_path import shortest_through_all_points


class UPBalancedKMeans:
    """
    Implements an UP-Balanced K-Means algorithm designed to cluster data points
    while considering associated probabilities and ensuring balanced cluster sizes.

    This algorithm first expands the input data points based on their probabilities,
    then applies a constrained K-Means algorithm to the expanded dataset. Finally,
    it maps the cluster assignments back to the original data points, determining
    their membership probabilities for each cluster.

    Args:
        k (int): The desired number of clusters.
        split_size (float, optional): A scaling factor used to expand the data points
                                      based on their probabilities. Smaller values lead
                                      to more expanded points. Defaults to 0.001.
    """

    def __init__(self, k: int, split_size: float = 0.001):
        self.k: int = k
        self.split_size: float = split_size
        self.N: Optional[int] = None  # Number of original data points
        self.membership: Optional[np.ndarray] = None  # Membership probabilities for each point to each cluster
        self.clusters: Optional[dict] = None 
        self.labels: Optional[np.ndarray] = None  # Final cluster labels for original points
        self.centroids: Optional[np.ndarray] = None  # Centroids for each cluster

    def _generate_expanded_coords(self, coords: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates expanded coordinates and their original indices based on probabilities.

        Each original data point is repeated a number of times proportional to its
        probability, scaled by `self.split_size`. This effectively gives more weight
        to points with higher probabilities during the constrained K-Means step.

        Args:
            coords (np.ndarray): A 2D NumPy array where each row is a coordinate of a data point.
            probs (np.ndarray): A 1D NumPy array of probabilities corresponding to each data point.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - expanded_coords (np.ndarray): The new array of expanded coordinates.
                - expanded_idx (np.ndarray): An array mapping each expanded coordinate back
                                            to its original index in `coords`.
                - original_point_expansion_counts (np.ndarray): The number of times each original
                                                                point was expanded.
        """
        # Calculate how many times each point should be repeated
        # Round to nearest integer and ensure at least one repeat
        original_point_expansion_counts: np.ndarray = (probs / self.split_size).round().astype(int)
        original_point_expansion_counts[original_point_expansion_counts == 0] = 1

        # Repeat the coordinates and their original indices based on calculated counts
        expanded_coords: np.ndarray = np.repeat(coords, original_point_expansion_counts, axis=0)
        expanded_idx: np.ndarray = np.repeat(np.arange(self.N), original_point_expansion_counts)

        return expanded_coords, expanded_idx, original_point_expansion_counts

    def _generate_membership(self, extended_labels: np.ndarray, expanded_idx: np.ndarray,
                             original_point_expansion_counts: np.ndarray) -> np.ndarray:
        """
        Calculates the membership probabilities for each original data point to each cluster.

        This is done by counting how many times each expanded representation of an original
        point was assigned to a particular cluster, and then normalizing these counts
        by the total number of expanded representations for that original point.

        Args:
            extended_labels (np.ndarray): Labels assigned to the expanded coordinates by KMeansConstrained.
            expanded_idx (np.ndarray): An array mapping each expanded coordinate back
                                       to its original index in the original dataset.
            original_point_expansion_counts (np.ndarray): The number of times each original
                                                          point was expanded.

        Returns:
            np.ndarray: A 2D NumPy array (N, k) where N is the number of original points
                        and k is the number of clusters. Each element (i, j) represents
                        the probability that original point 'i' belongs to cluster 'j'.
        """
        # Create a matrix to count how many times each expanded point (from original_idx)
        # was assigned to each cluster (extended_label)
        membership_counts_matrix: np.ndarray = np.zeros((self.N, self.k), dtype=int)
        np.add.at(membership_counts_matrix, (expanded_idx, extended_labels), 1)

        # Divide the counts by the total number of expanded points for each original point
        # This gives the proportion (membership probability)
        # Using np.newaxis to enable broadcasting for division
        membership: np.ndarray = membership_counts_matrix / original_point_expansion_counts[:, np.newaxis]
        return membership

    def _generate_clusters(
            self,
            order: np.ndarray,
            labels: np.ndarray,
            centroids: np.ndarray,
            coords: np.ndarray,
            probs: np.ndarray,
    ) -> dict:
        """
        Simple 'mega-order' implementation:

        1) For each label in `order`, sort its indices by:
           - interior labels: ascending by (d_prev - d_next)  → prev-closest first, next-closest last
           - first label:     ascending by (-d_next)          → far-from-next first, closest-to-next last
           - last label:      ascending by (d_prev)           → closest-to-prev first
           Then concatenate all these blocks to form a single 'mega' index order of length N.

        2) Compute cumsum of probs over this mega order, and find border positions at
           thresholds 1, 2, ..., k (leftmost indices where cumsum ≥ threshold).

        3) At border i (between clusters i-1 and i), split the border index so that:
           - `ceil` goes to cluster i-1 to make its sum exactly i,
           - `floor` (remainder) goes to cluster i.
           First cluster has no incoming floor; last cluster has no outgoing ceil.

        4) Save clusters:
           label: {
             'free': indices strictly between its incoming and outgoing borders,
             'border': {
               'floor_index', 'floor_percentage',  # incoming (none for first)
               'ceil_index',  'ceil_percentage'    # outgoing (none for last)
             }
           }
        """
        N = len(probs)
        eps = 1e-12

        # -- Step 1: build mega order ------------------------------------------------
        # Pre-gather indices per label
        label_to_indices = {lab: np.flatnonzero(labels == lab) for lab in order}

        mega_blocks = []
        for i, lab in enumerate(order):
            idx = label_to_indices.get(lab, np.array([], dtype=int))
            if idx.size == 0:
                mega_blocks.append(idx)
                continue

            # prev/next labels
            has_prev = (i > 0)
            has_next = (i < self.k - 1)
            if has_prev:
                prev_lab = int(order[i - 1])
                d_prev = np.linalg.norm(coords[idx] - centroids[prev_lab], axis=1)
            else:
                d_prev = None
            if has_next:
                next_lab = int(order[i + 1])
                d_next = np.linalg.norm(coords[idx] - centroids[next_lab], axis=1)
            else:
                d_next = None

            if has_prev and has_next:
                # ascending by (d_prev - d_next): prev-closest first, next-closest last
                key = d_prev - d_next
            elif not has_prev and has_next:
                # first cluster: far-from-next first (closest to next at the end)
                key = -d_next
            elif has_prev and not has_next:
                # last cluster: closest-to-prev first
                key = d_prev
            else:
                # k == 1 edge-case: keep as-is
                key = np.zeros(idx.shape[0])

            order_within = np.argsort(key, kind="stable")
            mega_blocks.append(idx[order_within])

        mega_order = np.concatenate(mega_blocks) if mega_blocks else np.array([], dtype=int)

        # -- Step 2: borders at thresholds 1..k -------------------------------------
        if N == 0:
            return {int(lab): {'free': [], 'border': {'floor_index': -1, 'floor_percentage': 0.0,
                                                      'ceil_index': -1, 'ceil_percentage': 0.0}}
                    for lab in order}

        p_sorted = probs[mega_order]
        cs = np.cumsum(p_sorted)

        probs_total = round(np.sum(probs))
        m = round(probs_total / self.k)

        thresholds = np.arange(m, probs_total + 1, step=m, dtype=float)  # 1,2,...,k
        border_pos = np.searchsorted(cs, thresholds, side='left')  # positions in mega_order (len k)

        # -- Step 3: split each border (except the last) ----------------------------
        # For border i (i=1..k-1):
        #   pos = border_pos[i-1]
        #   ceil% for cluster i-1 = (i - mass_before) / prob_at_pos
        #   floor% for cluster i   = 1 - ceil%
        pos_1_to_km1 = border_pos[:-1]  # positions for i=1..k-1
        mass_before = np.where(pos_1_to_km1 > 0, cs[pos_1_to_km1 - 1], 0.0)
        border_idx_at_pos = mega_order[pos_1_to_km1] if pos_1_to_km1.size else np.array([], dtype=int)
        border_prob = probs[border_idx_at_pos] if pos_1_to_km1.size else np.array([], dtype=float)
        need = thresholds[:-1] - mass_before

        with np.errstate(divide='ignore', invalid='ignore'):
            ceil_pct_for_prev = np.clip(need / np.maximum(border_prob, eps), 0.0, 1.0)
        # floor for current cluster is 1 - ceil of previous
        floor_pct_for_curr = 1.0 - ceil_pct_for_prev

        # -- Step 4: assemble per-cluster outputs -----------------------------------
        clusters = {}
        for c, lab in enumerate(order):
            # incoming floor (border at threshold c), except for first cluster
            if c == 0:
                floor_index = -1
                floor_percentage = 0.0
                start_pos = 0
            else:
                pos_in = int(border_pos[c - 1])  # position of border c in mega_order
                floor_index = int(mega_order[pos_in])
                floor_percentage = float(floor_pct_for_curr[c - 1])
                start_pos = pos_in + 1  # free region starts after incoming border

            # outgoing ceil (border at threshold c+1), except for last cluster
            if c == self.k - 1:
                ceil_index = -1
                ceil_percentage = 0.0
                end_pos = N  # free region ends at end
            else:
                pos_out = int(border_pos[c])  # position of border c+1
                ceil_index = int(mega_order[pos_out])
                ceil_percentage = float(ceil_pct_for_prev[c])
                end_pos = pos_out  # free region ends before outgoing border

            # free = strictly between incoming and outgoing borders in mega order
            if end_pos > start_pos:
                free_slice = mega_order[start_pos:end_pos]
                free_list = free_slice.tolist()
            else:
                free_list = []

            clusters[int(lab)] = {
                'free': np.array(free_list, dtype=int),
                'border': {
                    'floor_index': floor_index,
                    'floor_percentage': float(floor_percentage),
                    'ceil_index': ceil_index,
                    'ceil_percentage': float(ceil_percentage),
                }
            }

        return clusters

    def _generate_labels(self, clusters: dict, probs: np.ndarray) -> np.ndarray:
        """
        Build hard labels from cluster splits.
        Each point may contribute to up to two clusters (via a border split).
        We compute a per-point per-cluster fraction, then take argmax.

        Args:
            clusters: dict[label] -> {'free': [...], 'border': {...}}
            probs:    (N,) probabilities for each point

        Returns:
            labels: (N,) hard label per point (cluster id), or -1 if truly unassigned.
        """
        N = probs.shape[0]
        k = self.k
        # Fractional membership matrix (not normalized, but sums to <= 1 per row)
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

        # Hard labels = argmax over cluster fractions; keep -1 where zero across all
        labels = np.full(N, -1, dtype=int)
        best_lab = np.argmax(frac, axis=1)
        has_any = (frac.max(axis=1) > 0.0)
        labels[has_any] = best_lab[has_any]
        return labels

    def fit(self, population: Population) -> None:
        """
        Fits the Auxiliary Balanced K-Means model to the given population data.

        Args:
            population (Population): An instance of the Population class containing
                                     the coordinates, probabilities, and optional IDs
                                     of the data points.
        """
        # Extract data from the Population object
        coords: np.ndarray = population.coords
        probs: np.ndarray = population.probs

        self.N = population.N

        # Generate expanded coordinates based on probabilities
        expanded_coords, expanded_idx, original_point_expansion_counts = self._generate_expanded_coords(coords, probs)

        # Apply KMeansConstrained to the expanded dataset
        # Calculate ideal cluster size for the constrained K-Means
        # Ensure that cluster_size is at least 1 to avoid issues with KMeansConstrained
        cluster_size: int = max(1, len(expanded_idx) // self.k)

        # Initialize and fit KMeansConstrained
        # n_jobs=-1 uses all available CPU cores for parallel processing
        kmeans = KMeansConstrained(
            n_clusters=self.k,
            size_min=cluster_size,
            size_max=cluster_size+1 if self.k > 1 else cluster_size,  # Allows for slight variation in cluster sizes
            n_jobs=-1,
            random_state=42 # For reproducibility
        )
        # extended_labels are the cluster assignments for the expanded points
        extended_labels: np.ndarray = kmeans.fit_predict(expanded_coords)

        # Calculate membership probabilities for each original point to each cluster
        self.membership = self._generate_membership(extended_labels, expanded_idx, original_point_expansion_counts)

        # Map cluster assignments back to original data points
        # Determine the final labels for the original data points
        raw_labels: np.ndarray = np.argmax(self.membership, axis=1)

        # Compute centroids for each cluster based on original coordinates and labels
        raw_centroids: np.ndarray = np.array([coords[raw_labels == i].mean(axis=0) for i in range(self.k)])

        ordered_labels = shortest_through_all_points(raw_centroids)

        clusters = self._generate_clusters(ordered_labels, raw_labels, raw_centroids, coords, probs)

        self.labels = self._generate_labels(clusters, probs)
        self.centroids: np.ndarray = np.array([coords[ self.labels == i].mean(axis=0) for i in range(self.k)])

        self.clusters = list(clusters.values())
