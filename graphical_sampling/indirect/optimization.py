"""Linked neighbors and auxiliary criteria for the package GBFS engine."""
import math
import operator
import random

import numpy as np

from ..search import GreedyBestFirstSearch


def _integer(value, name, minimum):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer >= {minimum}') from exc
    if result < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return result


def optimize_gbfs(initial, iterations=150, proposals=35, alpha=.5,
                  fraction=.75, max_frontier=100, seed=2026, keep_best=10,
                  stagnation_limit=10, backtrack_depth=50):
    """Optimize a linked design with the existing GreedyBestFirstSearch.

    The objective is alpha*VA/VA0 + (1-alpha)*VB/VB0. An exactly zero
    baseline uses unit scale, so subsequent positive variance is penalized.
    This heuristic explores only discovered designs, without any global
    optimality guarantee. Full candidate distributions are retained in memory.

    ``max_frontier`` uses GBFS's existing pruning threshold: the queue is
    reduced to this size whenever it grows beyond twice this value.
    """
    iterations = _integer(iterations, 'iterations', 0)
    proposals = _integer(proposals, 'proposals', 1)
    max_frontier = _integer(max_frontier, 'max_frontier', 1)
    keep_best = _integer(keep_best, 'keep_best', 1)
    stagnation_limit = _integer(stagnation_limit, 'stagnation_limit', 1)
    backtrack_depth = _integer(backtrack_depth, 'backtrack_depth', 0)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    initial.validate()
    rng = np.random.default_rng(seed)
    va0, vb0 = initial.nht_variance(initial.cfg.x_A, initial.cfg.x_B)
    scales = (va0 if va0 > 0 else 1., vb0 if vb0 > 0 else 1.)

    def score(design):
        va, vb = design.nht_variance(design.cfg.x_A, design.cfg.x_B)
        value = alpha*va/scales[0] + (1-alpha)*vb/scales[1]
        if not math.isfinite(value):
            raise ArithmeticError('Nonfinite auxiliary variance objective')
        return value

    def neighbors(design, iteration):
        states = list(design.states)
        if len(states) < 2:
            return
        weights = np.array(list(design.states.values()))
        weights /= weights.sum()
        for _ in range(proposals):
            a, b = rng.choice(len(states), 2, replace=False, p=weights)
            u, v = states[a], states[b]
            e, f = u[int(rng.integers(len(u)))], v[int(rng.integers(len(v)))]
            candidate = design.exchange(u, v, e, f, fraction=fraction)
            if candidate is not None:
                yield 'linked_exchange', candidate

    candidates, candidate_iterations, history = [], [], []
    known = set()

    def observe(design, value, iteration):
        key = design.key()
        if key not in known:
            known.add(key)
            candidates.append((float(value), design))
            candidate_iterations.append(iteration)

    def progress(search, iteration):
        best = search.reservoir_design
        va, vb = best.nht_variance(best.cfg.x_A, best.cfg.x_B)
        history.append(dict(iteration=iteration, best_score=search.reservoir_val,
                            best_variance_A=va, best_variance_B=vb,
                            frontier_size=len(search.open_set),
                            accepted=max(0, len(candidates)-1)))

    search = GreedyBestFirstSearch([initial], score, state_key=lambda d: d.key(),
                                   neighbor_factory=neighbors,
                                   candidate_callback=observe, rng=random.Random(seed))
    observe(initial, score(initial), 0)
    progress(search, 0)
    history[0]['frontier_size'] = 1
    search.run(max_iterations=iterations, max_open_set_size=max_frontier,
               top_k=keep_best, num_new_order_nodes=0,
               num_new_exchange_nodes=proposals, num_clusters_range=(1, 1),
               num_zones_range=(1, 1), num_changes_range=(1, 1),
               num_zone_changes=0, n_jobs=1, output_mode='silent',
               stagnation_limit=stagnation_limit, backtrack_depth=backtrack_depth,
               progress_callback=progress)
    # The reservoir uses the package's nine-decimal improvement policy. Return
    # its actual state and value rather than claiming finer precision of search.
    return dict(initial=initial, best=search.best_design, search=search,
                initial_score=score(initial), best_score=search.best_criteria_value,
                history=history, candidate_designs=candidates,
                candidate_iterations=candidate_iterations,
                best_candidates=sorted(candidates, key=lambda item: item[0])[:keep_best],
                generated_candidates=len(candidates), explored_states=len(search.closed_set),
                alpha=alpha, scales=scales, seed=seed)
