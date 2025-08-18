import numpy as np
from k_means_constrained import KMeansConstrained
from typing import Optional, List, Tuple, Union, Dict
from ..population import Population
from .shortest_path import shortest_through_all_points


def _parse_index_share(arg: Union[bool, None, np.ndarray], N: int) -> Optional[Tuple[int, float]]:
    """
    Accepts False/None or a 2-vector [index, share]. Returns (idx, share) or None.
    Clips share to [0,1] and ignores out-of-range idx or nonpositive share.
    """
    if arg is False or arg is None:
        return None
    arr = np.asarray(arg, dtype=float)
    if arr.shape[0] != 2:
        return None
    idx = int(arr[0])
    if idx < 0 or idx >= N:
        return None
    share = float(arr[1])
    if not np.isfinite(share):
        return None
    share = float(np.clip(share, 0.0, 1.0))
    if share <= 0.0:
        return None
    return idx, share


class NestedUPBalancedKMeans:
    """
    UP-Balanced K-Means with border-aware slicing.

    New:
      - Optional `index_share_floor` and `index_share_ceil`, each either False/None
        or np.array([idx, share]). Only the mass `share * probs[idx]` is counted in
        this pass. The floor chunk is forced to the first cluster as its incoming
        'floor' border; the ceil chunk is forced to the last cluster as its outgoing
        'ceil' border. Both are excluded from 'free'.
      - `shortest_through_all_points` is called with `start`/`end` set to the labels
        of those indices (if provided), so the order starts at the floor’s label and
        ends at the ceil’s label.
    """

    def __init__(self, k: int, split_size: float = 0.001):
        self.k: int = k
        self.split_size: float = split_size
        self.N: Optional[int] = None
        self.membership: Optional[np.ndarray] = None
        self.clusters: Optional[dict] = None
        self.labels: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None

    def _generate_expanded_coords(
            self,
            coords: np.ndarray,
            probs: np.ndarray,
            index_share_floor: Optional[Tuple[int, float]] = None,
            index_share_ceil: Optional[Tuple[int, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Expand points proportionally to *effective* probabilities.
        If floor/ceil shares are provided, only share*prob is used for those indices
        in this pass (the remainder is intentionally not expanded here).

        Returns:
            expanded_coords, expanded_idx, original_point_expansion_counts
        """
        # start from full prob, then downscale special indices
        eff_probs = probs.astype(float).copy()

        # apply floor share
        if index_share_floor is not None:
            fi, fshare = int(index_share_floor[0]), float(index_share_floor[1])
            eff_probs[fi] = fshare * probs[fi]

        # apply ceil share (avoid double-count if same index appears twice)
        if index_share_ceil is not None:
            ci, cshare = int(index_share_ceil[0]), float(index_share_ceil[1])
            if index_share_floor is not None and ci == int(index_share_floor[0]):
                # if both point to same index, keep the larger share for this pass
                eff_probs[ci] = max(eff_probs[ci], cshare * probs[ci])
            else:
                eff_probs[ci] = cshare * probs[ci]

        # compute expansion counts from effective probabilities
        original_point_expansion_counts: np.ndarray = (eff_probs / self.split_size).round().astype(int)
        original_point_expansion_counts[
            original_point_expansion_counts == 0] = 1  # keep every point represented at least once

        expanded_coords: np.ndarray = np.repeat(coords, original_point_expansion_counts, axis=0)
        expanded_idx: np.ndarray = np.repeat(np.arange(self.N), original_point_expansion_counts)

        return expanded_coords, expanded_idx, original_point_expansion_counts

    def _generate_membership(self, extended_labels: np.ndarray, expanded_idx: np.ndarray,
                             original_point_expansion_counts: np.ndarray) -> np.ndarray:
        membership_counts_matrix: np.ndarray = np.zeros((self.N, self.k), dtype=int)
        np.add.at(membership_counts_matrix, (expanded_idx, extended_labels), 1)
        membership: np.ndarray = membership_counts_matrix / original_point_expansion_counts[:, np.newaxis]
        return membership

    def _generate_clusters(
        self,
        order: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        coords: np.ndarray,
        probs: np.ndarray,
        index_share_floor: Optional[Tuple[int, float]] = None,
        index_share_ceil: Optional[Tuple[int, float]] = None,
    ) -> dict:
        """
        Mega-order with optional fixed floor/ceil chunks:
          • Place floor index (if any) at the very beginning, counting only share*prob.
          • Place ceil index (if any) at the very end,   counting only share*prob.
          • Exclude those indices from interior blocks so they don't double-count.
          • Thresholds are set on the *effective* total mass of this pass
            (sum of p_sorted), so they work even when only partial mass of floor/ceil
            is considered in this pass.
        """
        N = len(probs)
        eps = 1e-12

        # -- Pre-gather indices per label (exclude special indices from blocks) -----
        excl = set()
        if index_share_floor is not None:
            excl.add(int(index_share_floor[0]))
        if index_share_ceil is not None:
            excl.add(int(index_share_ceil[0]))

        label_to_indices: Dict[int, np.ndarray] = {
            int(lab): np.setdiff1d(np.flatnonzero(labels == lab), np.fromiter(excl, dtype=int), assume_unique=False)
            for lab in order
        }

        # -- Build per-label blocks (sorting rule as before) ------------------------
        mega_blocks: List[np.ndarray] = []
        for i, lab in enumerate(order):
            idx = label_to_indices.get(int(lab), np.array([], dtype=int))
            if idx.size == 0:
                mega_blocks.append(idx)
                continue

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
                key = d_prev - d_next
            elif not has_prev and has_next:
                key = -d_next
            elif has_prev and not has_next:
                key = d_prev
            else:
                key = np.zeros(idx.shape[0])

            order_within = np.argsort(key, kind="stable")
            mega_blocks.append(idx[order_within])

        # -- Stitch mega order with explicit floor first and ceil last --------------
        pairs: List[Tuple[int, float]] = []

        # floor chunk first (if any)
        if index_share_floor is not None:
            fi, fshare = int(index_share_floor[0]), float(index_share_floor[1])
            pairs.append((fi, fshare * float(probs[fi])))

        # interior blocks (full mass)
        for blk in mega_blocks:
            for iidx in blk:
                pairs.append((int(iidx), float(probs[int(iidx)])))

        # ceil chunk last (if any)
        if index_share_ceil is not None:
            ci, cshare = int(index_share_ceil[0]), float(index_share_ceil[1])
            pairs.append((ci, cshare * float(probs[ci])))

        if not pairs:
            # no mass — return empty shells
            return {
                int(lab): {
                    'free': np.array([], dtype=int),
                    'border': {'floor_index': -1, 'floor_percentage': 0.0,
                               'ceil_index': -1, 'ceil_percentage': 0.0}
                } for lab in order
            }

        mega_order = np.array([p[0] for p in pairs], dtype=int)
        p_sorted   = np.array([p[1] for p in pairs], dtype=float)

        # -- Thresholds on effective mass of this pass ------------------------------
        cs = np.cumsum(p_sorted)
        S_eff = float(cs[-1])           # total effective mass being assigned in this pass
        m_eff = S_eff / float(self.k)   # target per cluster (not rounded; supports m=1 or any real)

        # thresholds: m, 2m, ..., k*m (length k)
        thresholds = (np.arange(1, self.k + 1, dtype=float) * m_eff)

        # leftmost positions where cumsum >= threshold
        border_pos = np.searchsorted(cs, thresholds, side='left')

        # -- Split each internal border (1..k-1) -----------------------------------
        pos_1_to_km1 = border_pos[:-1]                 # positions for i=1..k-1
        mass_before = np.where(pos_1_to_km1 > 0, cs[pos_1_to_km1 - 1], 0.0)
        border_idx_at_pos = mega_order[pos_1_to_km1] if pos_1_to_km1.size else np.array([], dtype=int)
        # use FULL prob in denominator → yields fraction of the point's total prob
        border_prob_full = probs[border_idx_at_pos] if pos_1_to_km1.size else np.array([], dtype=float)
        need = thresholds[:-1] - mass_before

        with np.errstate(divide='ignore', invalid='ignore'):
            ceil_pct_for_prev = np.clip(need / np.maximum(border_prob_full, eps), 0.0, 1.0)
        floor_pct_for_curr = 1.0 - ceil_pct_for_prev

        # -- Assemble clusters ------------------------------------------------------
        clusters: Dict[int, dict] = {}
        first_lab = int(order[0])
        last_lab  = int(order[-1])

        for c, lab in enumerate(order):
            lab = int(lab)
            # incoming floor (border at threshold c), except for first cluster
            if c == 0:
                floor_index = -1
                floor_percentage = 0.0
                # free region starts at 1 if we injected a floor chunk at position 0
                has_injected_floor = (index_share_floor is not None)
                start_pos = 1 if has_injected_floor else 0
            else:
                pos_in = int(border_pos[c - 1])
                floor_index = int(mega_order[pos_in])
                floor_percentage = float(floor_pct_for_curr[c - 1])
                start_pos = pos_in + 1

            # outgoing ceil (border at threshold c+1), except for last cluster
            if c == self.k - 1:
                ceil_index = -1
                ceil_percentage = 0.0
                end_pos = mega_order.shape[0]
            else:
                pos_out = int(border_pos[c])
                ceil_index = int(mega_order[pos_out])
                ceil_percentage = float(ceil_pct_for_prev[c])
                end_pos = pos_out

            # free = strictly between incoming and outgoing borders in mega order
            if end_pos > start_pos:
                free_slice = mega_order[start_pos:end_pos]
                free_list = free_slice.tolist()
            else:
                free_list = []

            clusters[lab] = {
                'free': np.array(free_list, dtype=int),
                'border': {
                    'floor_index': floor_index,
                    'floor_percentage': float(floor_percentage),
                    'ceil_index': ceil_index,
                    'ceil_percentage': float(ceil_percentage),
                }
            }

        # -- Inject fixed floor/ceil into first/last clusters and scrub from free ---
        if index_share_floor is not None:
            fi, fshare = int(index_share_floor[0]), float(index_share_floor[1])
            clusters[first_lab]['border']['floor_index'] = fi
            clusters[first_lab]['border']['floor_percentage'] = fshare
            # remove from any free sets (esp. first cluster)
            for lab in clusters.keys():
                clusters[lab]['free'] = clusters[lab]['free'][clusters[lab]['free'] != fi]

        if index_share_ceil is not None:
            ci, cshare = int(index_share_ceil[0]), float(index_share_ceil[1])
            clusters[last_lab]['border']['ceil_index'] = ci
            clusters[last_lab]['border']['ceil_percentage'] = cshare
            # remove from any free sets (esp. last cluster)
            for lab in clusters.keys():
                clusters[lab]['free'] = clusters[lab]['free'][clusters[lab]['free'] != ci]

        return clusters

    def _generate_labels(self, clusters: dict, probs: np.ndarray) -> np.ndarray:
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

    def fit(
            self,
            population: Population,
            index_share_floor: Union[bool, None, np.ndarray] = False,
            index_share_ceil: Union[bool, None, np.ndarray] = False,
    ) -> None:
        coords: np.ndarray = population.coords
        probs: np.ndarray = population.probs
        self.N = population.N

        # parse special indices now (so expansion sees the reduced mass)
        floor_info = _parse_index_share(index_share_floor, self.N)
        ceil_info = _parse_index_share(index_share_ceil, self.N)

        # use effective probs for expansion
        expanded_coords, expanded_idx, expansion_counts = self._generate_expanded_coords(
            coords, probs, index_share_floor=floor_info, index_share_ceil=ceil_info
        )

        cluster_size: int = max(1, len(expanded_idx) // self.k)
        kmeans = KMeansConstrained(
            n_clusters=self.k,
            size_min=cluster_size,
            size_max=cluster_size + 1 if self.k > 1 else cluster_size,
            n_jobs=-1,
            # random_state=42
        )
        extended_labels: np.ndarray = kmeans.fit_predict(expanded_coords)
        self.membership = self._generate_membership(extended_labels, expanded_idx, expansion_counts)

        raw_labels: np.ndarray = np.argmax(self.membership, axis=1)
        raw_centroids: np.ndarray = np.array([coords[raw_labels == i].mean(axis=0) for i in range(self.k)])

        # seed order with the clusters containing the floor/ceil points
        start_lab = int(raw_labels[floor_info[0]]) if floor_info is not None else None
        end_lab = int(raw_labels[ceil_info[0]]) if ceil_info is not None else None
        order_labels = shortest_through_all_points(raw_centroids, start=start_lab, end=end_lab)

        # build clusters using partial masses for floor/ceil in the cumsum logic
        self.clusters = self._generate_clusters(
            order_labels, raw_labels, raw_centroids, coords, probs,
            index_share_floor=floor_info,
            index_share_ceil=ceil_info
        )

        self.labels = self._generate_labels(self.clusters, probs)
        self.centroids = np.array([coords[self.labels == i].mean(axis=0) for i in range(self.k)])
