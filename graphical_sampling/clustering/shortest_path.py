import numpy as np


def _pairwise_dist(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))


def _path_length_open(order: np.ndarray, D: np.ndarray) -> float:
    if len(order) <= 1:
        return 0.0
    return float(D[order[:-1], order[1:]].sum())


def _nearest_neighbor_open(D: np.ndarray, start: int, end: int | None = None) -> np.ndarray:
    """
    Nearest-neighbor for OPEN path, starting at `start`.
    If `end` is given and end != start (unless n==1), it will be the last node.
    """
    n = D.shape[0]
    unvisited = np.ones(n, dtype=bool)
    order = np.empty(n, dtype=int)

    order[0] = start
    unvisited[start] = False

    # If we must end at `end`, forbid choosing it until the last step.
    lock_end = end is not None and n > 1
    if lock_end:
        if not (0 <= end < n):
            raise ValueError(f"end index {end} out of range [0,{n})")
        if end == start and n > 1:
            raise ValueError("start and end cannot be the same when n>1.")
        # keep end unvisited but masked out until final selection
        # (still marked unvisited so it can be picked at the last step)
        pass

    last = start
    for t in range(1, n):
        drow = D[last].copy()
        # mask out already visited
        drow[~unvisited] = np.inf
        # additionally mask out `end` until the last position
        if lock_end and t < n - 1:
            drow[end] = np.inf
        nxt = int(np.argmin(drow))
        order[t] = nxt
        unvisited[nxt] = False
        last = nxt

    # If locked, ensure we indeed ended at `end` (should be guaranteed)
    if lock_end and order[-1] != end:
        # Safety fallback (should rarely trigger): swap last with position of `end`
        pos_end = int(np.nonzero(order == end)[0][0])
        order[pos_end], order[-1] = order[-1], order[pos_end]
    return order


def _precompute_2opt_indices_open(n: int):
    # Valid (i,k): 0 <= i < k-1 <= n-2  => i in [0, n-3], k in [i+2, n-1] and k < n-1
    I, K = np.triu_indices(n, k=2)
    mask = (K < n - 1)  # ensure k+1 exists
    return I[mask], K[mask]


def _two_opt_open(order: np.ndarray, D: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """
    Best-improvement 2-opt for OPEN path.
    Endpoints order[0] and order[-1] remain fixed (segment reversal is internal).
    """
    n = len(order)
    if n < 4:
        return order

    I, K = _precompute_2opt_indices_open(n)

    for _ in range(max_iter):
        a = order[I]
        b = order[I + 1]
        c = order[K]
        d = order[K + 1]

        old = D[a, b] + D[c, d]
        new = D[a, c] + D[b, d]
        gain = old - new

        best_idx = int(np.argmax(gain))
        if gain[best_idx] > 1e-12:
            i = int(I[best_idx])
            k = int(K[best_idx])
            order[i + 1 : k + 1] = order[i + 1 : k + 1][::-1]
        else:
            break
    return order


def shortest_through_all_points(points, *, restarts=8, seed=42,
                                max_two_opt_iters=1000, start=None, end=None):
    """
    Shortest OPEN path through all points (NumPy-accelerated).

    Args:
        points: iterable of (x, y)
        restarts: NN+2opt restarts with different starts (ignored if `start` given)
        seed: RNG seed
        max_two_opt_iters: max 2-opt iterations per restart
        start: index to start from; if None, try multiple starts
        end:   index to end at; if None, unconstrained end. If provided,
               the returned path always ends at this index.

    Returns:
        order (np.ndarray of indices), length (float)
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n == 0:
        return np.array([], dtype=int), 0.0
    if n == 1:
        if start is not None and start != 0:
            raise ValueError("For n=1, only valid start is 0.")
        if end is not None and end != 0:
            raise ValueError("For n=1, only valid end is 0.")
        return np.array([0], dtype=int), 0.0

    D = _pairwise_dist(P)

    # Validate provided indices
    if start is not None and not (0 <= start < n):
        raise ValueError(f"start index {start} out of range [0,{n})")
    if end is not None and not (0 <= end < n):
        raise ValueError(f"end index {end} out of range [0,{n})")
    if start is not None and end is not None and n > 1 and start == end:
        raise ValueError("start and end cannot be the same when n>1.")

    if start is not None:
        starts = [start]
    else:
        rng = np.random.default_rng(seed)
        starts = set(rng.integers(0, n, size=min(restarts, n)).tolist())
        # also try farthest-from-centroid as a deterministic seed
        centroid = P.mean(axis=0)
        far_idx = int(np.argmax(np.einsum('ij,ij->i', P - centroid, P - centroid)))
        starts.add(far_idx)
        # avoid choosing `end` as a start if end is fixed and n>1
        if end is not None and end in starts and n > 1:
            starts.discard(end)
            # if we removed too many, add a random alternative
            while len(starts) == 0:
                starts.add(int(rng.integers(0, n)))
                if end in starts:
                    starts.discard(end)
        starts = list(starts)[:min(restarts, n if end is None else max(1, n - 1))]

    best_order = None
    best_len = np.inf

    for s in starts:
        order = _nearest_neighbor_open(D, s, end=end)
        order = _two_opt_open(order, D, max_iter=max_two_opt_iters)
        # Start and end are preserved by construction; no rotation needed.
        L = _path_length_open(order, D)
        if L < best_len:
            best_len = L
            best_order = order.copy()

    return best_order
