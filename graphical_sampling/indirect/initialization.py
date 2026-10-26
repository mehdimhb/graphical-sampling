"""Transportation flow and augmented Birkhoff matching decomposition."""
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .design import TOL


def feasible_link_masses(cfg):
    """Find nonnegative allowed link masses with the prescribed FIP margins.

    Feasibility depends on the complete link network, not just whether each
    unit has a neighbor. A violated flow constraint raises ValueError.
    """
    aa, bb = np.where(cfg.links)
    count = len(aa)
    cols = np.tile(np.arange(count), 2)
    rows = np.r_[aa, len(cfg.pi_A) + bb]
    matrix = coo_matrix((np.ones(2*count), (rows, cols)),
                        shape=(len(cfg.pi_A)+len(cfg.pi_B), count)).tocsr()
    result = linprog(np.zeros(count), A_eq=matrix,
                     b_eq=np.r_[cfg.pi_A, cfg.pi_B], bounds=(0, None),
                     method='highs', options={'primal_feasibility_tolerance': 1e-10})
    if not result.success:
        raise ValueError(f'Infeasible linkage / FIP marginals: {result.message}')
    lm = np.zeros(cfg.links.shape)
    lm[aa, bb] = result.x
    if (not np.allclose(lm.sum(axis=1), cfg.pi_A, atol=TOL, rtol=0)
            or not np.allclose(lm.sum(axis=0), cfg.pi_B, atol=TOL, rtol=0)
            or np.any(lm.sum(axis=1) == 0) or np.any(lm.sum(axis=0) == 0)):
        raise ArithmeticError('LP margins inaccurate; FIPs may be below solver resolution')
    return lm


def _fill_transport(rows, columns):
    rr, cc = np.maximum(rows, 0).copy(), np.maximum(columns, 0).copy()
    arr = np.zeros((len(rr), len(cc)))
    i = j = 0
    while i < len(rr) and j < len(cc):
        value = min(rr[i], cc[j])
        arr[i, j] = value
        rr[i] -= value
        cc[j] -= value
        if rr[i] == 0:
            i += 1
        if cc[j] == 0:
            j += 1
    if np.any(rr > TOL) or np.any(cc > TOL):
        raise ArithmeticError('Unable to construct augmented matrix')
    return arr


def _decompose(masses, n):
    """Zero dummy/dummy block forces n real edges in every permutation."""
    na, nb = masses.shape
    size = na + nb - n
    residual = np.zeros((size, size))
    residual[:na, :nb] = masses
    if na > n:
        residual[:na, nb:] = _fill_transport(1-masses.sum(axis=1), np.ones(na-n))
    if nb > n:
        residual[na:, :nb] = _fill_transport(np.ones(nb-n), 1-masses.sum(axis=0))
    if (not np.allclose(residual.sum(axis=0), 1, atol=TOL, rtol=0)
            or not np.allclose(residual.sum(axis=1), 1, atol=TOL, rtol=0)):
        raise ArithmeticError('Augmented matrix is not doubly stochastic')
    out = defaultdict(float)
    numerical_zero = 32 * np.finfo(float).eps * size
    for _ in range(np.count_nonzero(residual) + 1):
        if residual.max() <= numerical_zero:
            break
        matching = maximum_bipartite_matching(csr_matrix(residual > 0), perm_type='column')
        if np.any(matching < 0):
            raise ArithmeticError('No perfect matching in residual support; numerical decomposition failed')
        amount = float(residual[np.arange(size), matching].min())
        edges = tuple((i, int(k)) for i, k in enumerate(matching[:na]) if k < nb)
        if len(edges) != n or amount <= 0:
            raise ArithmeticError('Invalid augmented matching decomposition')
        out[edges] += amount
        residual[np.arange(size), matching] -= amount
    else:
        raise ArithmeticError('Matching decomposition exceeded iteration budget')
    if residual.max() > numerical_zero or abs(sum(out.values()) - 1) > TOL:
        raise ArithmeticError('Matching decomposition did not exhaust probability mass')
    return dict(out)
