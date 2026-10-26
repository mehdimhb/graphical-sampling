"""Finite distributions over fixed-size bipartite matchings."""
from __future__ import annotations

import math
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

TOL = 1e-9


def _vector(value, size, name):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(f'{name} must be a finite vector of length {size}')
    return arr


def _frozen_array(value):
    arr = np.asarray(value)
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


@dataclass(frozen=True, eq=False)
class OGISInputs:
    """Two populations and prescribed positive inclusion probabilities.

    Unit identities are zero-based indices; names are display labels only.
    Study variables are optional simulation inputs and never used for scoring.
    Arrays are copied and read-only so later caller edits cannot alter a run.
    """

    pi_A: np.ndarray
    pi_B: np.ndarray
    links: np.ndarray
    x_A: np.ndarray
    x_B: np.ndarray
    y_A: np.ndarray | None = None
    y_B: np.ndarray | None = None
    names_A: tuple | None = None
    names_B: tuple | None = None
    n: int = field(init=False)

    def __post_init__(self):
        for name in ('pi_A', 'pi_B'):
            arr = np.asarray(getattr(self, name), dtype=float)
            if (arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all()
                    or np.any(arr <= 0) or np.any(arr > 1)):
                raise ValueError(f'{name} must be a nonempty finite vector in (0, 1]')
            object.__setattr__(self, name, _frozen_array(arr))
        na, nb = len(self.pi_A), len(self.pi_B)
        n = round(math.fsum(self.pi_A))
        if (not 1 <= n <= min(na, nb)
                or abs(math.fsum(self.pi_A) - n) > TOL
                or abs(math.fsum(self.pi_B) - n) > TOL):
            raise ValueError('Both FIP vectors must sum to the same positive integer n')
        object.__setattr__(self, 'n', n)
        links = np.asarray(self.links)
        if links.shape != (na, nb) or not np.isin(links, [0, 1]).all():
            raise ValueError(f'links must be a binary matrix of shape {(na, nb)}')
        links = links.astype(bool)
        if not links.any(axis=0).all() or not links.any(axis=1).all():
            raise ValueError('Every unit with positive FIP must have an allowed link')
        object.__setattr__(self, 'links', _frozen_array(links))
        for suffix, size in [('A', na), ('B', nb)]:
            for prefix in ('x', 'y'):
                name = f'{prefix}_{suffix}'
                value = getattr(self, name)
                if value is None and prefix == 'y':
                    continue
                arr = _vector(value, size, name)
                object.__setattr__(self, name, _frozen_array(arr))
            name = f'names_{suffix}'
            labels = getattr(self, name)
            labels = tuple(f'{suffix}{i+1}' for i in range(size)) if labels is None else tuple(labels)
            if len(labels) != size or any(not isinstance(s, str) for s in labels):
                raise ValueError(f'{name} must contain {size} string labels')
            object.__setattr__(self, name, labels)


def _pairs_key(edges):
    try:
        return tuple(sorted((operator.index(i), operator.index(k)) for i, k in edges))
    except (ValueError, TypeError) as exc:
        raise ValueError('Each linked pair must contain two integer indices') from exc


class LinkedDesign:
    """An immutable support distribution with a fixed link-mass matrix.

    ``states`` maps tuples of (A index, B index) pairs to strip probabilities.
    Zero masses may be omitted; negative and nonfinite masses are errors.
    """

    def __init__(self, cfg: OGISInputs, states, fixed_lambda=None):
        self._cfg = cfg
        fragments = defaultdict(list)
        for state, probability in states.items():
            p = float(probability)
            if not math.isfinite(p) or p < 0:
                raise ValueError('Strip probabilities must be finite and nonnegative')
            if p:
                fragments[_pairs_key(state)].append(p)
        self._states = MappingProxyType({e: math.fsum(ps) for e, ps in sorted(fragments.items())})
        lm = self.link_masses if fixed_lambda is None else np.asarray(fixed_lambda, dtype=float)
        self._fixed_lambda = _frozen_array(lm)
        self.validate()

    @property
    def cfg(self):
        return self._cfg

    @property
    def states(self):
        return self._states

    @property
    def fixed_lambda(self):
        return self._fixed_lambda

    @classmethod
    def initialize(cls, cfg):
        from .initialization import feasible_link_masses, _decompose

        lm = feasible_link_masses(cfg)
        return cls(cfg, _decompose(lm, cfg.n), fixed_lambda=lm)

    @property
    def link_masses(self):
        arr = np.zeros(self.cfg.links.shape)
        for edges, p in self.states.items():
            for i, k in edges:
                if not 0 <= i < arr.shape[0] or not 0 <= k < arr.shape[1]:
                    raise ValueError('Matching includes an out-of-range link')
                arr[i, k] += p
        return arr

    @property
    def conditional_probabilities(self):
        """q_ik = lambda_ik / pi_A[i], conditional on selecting A_i."""
        return self.link_masses / self.cfg.pi_A[:, None]

    def validate(self):
        cfg = self.cfg
        if not self.states or abs(math.fsum(self.states.values()) - 1) > TOL:
            raise ValueError('Strip probabilities must sum to one')
        for edges in self.states:
            if len(edges) != cfg.n:
                raise ValueError('Every strip must contain exactly n linked pairs')
            if len({i for i, _ in edges}) != cfg.n or len({k for _, k in edges}) != cfg.n:
                raise ValueError('Duplicate A or B selection in a matching')
            for i, k in edges:
                if (not 0 <= i < len(cfg.pi_A) or not 0 <= k < len(cfg.pi_B)
                        or not cfg.links[i, k]):
                    raise ValueError('Matching includes a forbidden link')
        lm = self.link_masses
        if (not np.allclose(lm.sum(axis=1), cfg.pi_A, atol=TOL, rtol=0)
                or not np.allclose(lm.sum(axis=0), cfg.pi_B, atol=TOL, rtol=0)):
            raise ValueError('FIP constraints violated')
        if (self.fixed_lambda.shape != lm.shape or not np.isfinite(self.fixed_lambda).all()
                or np.any(self.fixed_lambda < 0)
                or not np.allclose(lm, self.fixed_lambda, atol=TOL, rtol=0)):
            raise ValueError('Fixed link mass constraints violated')
        return True

    def key(self):
        """Probability-aware immutable snapshot without rounding away small strips."""
        return tuple(self.states.items())

    def exchange(self, u, v, e, f, fraction=.75):
        """Move intact linked pairs between equal-mass substrips.

        Return None when the proposed swap is forbidden or leaves the design
        unchanged. No endpoint is reassigned or deduplicated after selection.
        """
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('fraction must lie in (0, 1]')
        u, v, e, f = _pairs_key(u), _pairs_key(v), _pairs_key([e])[0], _pairs_key([f])[0]
        if u == v or u not in self.states or v not in self.states or e not in u or f not in v or e == f:
            return None
        nu = tuple(sorted([z for z in u if z != e] + [f]))
        nv = tuple(sorted([z for z in v if z != f] + [e]))
        for edges in (nu, nv):
            if len({i for i, _ in edges}) != self.cfg.n or len({k for _, k in edges}) != self.cfg.n:
                return None
        delta = fraction * min(self.states[u], self.states[v])
        states = dict(self.states)
        states[u] -= delta
        states[v] -= delta
        states[nu] = states.get(nu, 0) + delta
        states[nv] = states.get(nv, 0) + delta
        result = LinkedDesign(self.cfg, states, self.fixed_lambda)
        return None if result.key() == self.key() else result

    def sip(self):
        sa = np.zeros((len(self.cfg.pi_A), len(self.cfg.pi_A)))
        sb = np.zeros((len(self.cfg.pi_B), len(self.cfg.pi_B)))
        for edges, p in self.states.items():
            a, b = np.array(edges).T
            sa[np.ix_(a, a)] += p
            sb[np.ix_(b, b)] += p
        return sa, sb

    def nht_variance(self, values_A, values_B):
        """Exact design variances of NHT total estimators for A and B."""
        a = _vector(values_A, len(self.cfg.pi_A), 'values_A')
        b = _vector(values_B, len(self.cfg.pi_B), 'values_B')
        total_a, total_b = math.fsum(a), math.fsum(b)
        va, vb = [], []
        for edges, p in self.states.items():
            ah = math.fsum(a[i] / self.cfg.pi_A[i] for i, _ in edges)
            bh = math.fsum(b[k] / self.cfg.pi_B[k] for _, k in edges)
            va.append(p * (ah - total_a)**2)
            vb.append(p * (bh - total_b)**2)
        return math.fsum(va), math.fsum(vb)

    def draw(self, seed=None):
        """Draw one matching, using an integer seed or a NumPy Generator."""
        rng = np.random.default_rng(seed)
        keys = list(self.states)
        probs = np.array(list(self.states.values()))
        return keys[int(rng.choice(len(keys), p=probs / probs.sum()))]

    def sample_table(self):
        return [dict(probability=p,
                     links='; '.join(f'{self.cfg.names_A[i]}->{self.cfg.names_B[k]}' for i, k in edges),
                     sample_A=', '.join(self.cfg.names_A[i] for i, _ in edges),
                     sample_B=', '.join(self.cfg.names_B[k] for _, k in edges))
                for edges, p in self.states.items()]
