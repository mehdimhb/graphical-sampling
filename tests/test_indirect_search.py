"""OGIS uses the package search while preserving matching constraints."""
import random

import numpy as np
import pytest

from graphical_sampling.indirect import OGISInputs, LinkedDesign, optimize_gbfs
from graphical_sampling.search import GreedyBestFirstSearch
from test_indirect_design import paper_design, paper_inputs


def test_package_gbfs_integration_and_seed_isolation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Linked search must bypass legacy order/exchange operators')

    monkeypatch.setattr(GreedyBestFirstSearch, '_apply_exchange', forbidden)
    monkeypatch.setattr(GreedyBestFirstSearch, '_apply_order_change', forbidden)
    initial = paper_design()
    py_state = random.getstate()
    np_state = np.random.get_state()
    first = optimize_gbfs(initial, iterations=30, proposals=30, seed=17)
    assert isinstance(first['search'], GreedyBestFirstSearch)
    assert first['best_score'] < first['initial_score']
    second = optimize_gbfs(initial, iterations=30, proposals=30, seed=17)
    assert first['best'].key() == second['best'].key()
    assert first['history'] == second['history']
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert np_state[0] == after[0] and np_state[2:] == after[2:]
    np.testing.assert_array_equal(np_state[1], after[1])
    assert first['generated_candidates'] == len(first['candidate_designs'])
    assert first['candidate_designs'][0][1] is initial
    for score, d in first['candidate_designs']:
        assert d.validate() and np.isfinite(score)
        np.testing.assert_allclose(d.link_masses, initial.link_masses, atol=1e-12, rtol=0)
    assert first['search'].best_design is first['best']
    assert first['search'].reservoir_val == first['best_score']


def test_tiny_frontier_restart_does_not_lose_reservoir():
    result = optimize_gbfs(paper_design(), iterations=12, proposals=20,
                           max_frontier=1, stagnation_limit=1,
                           backtrack_depth=50, seed=8)
    assert result['best_score'] <= result['initial_score']
    assert result['best_score'] == min(s for s, _ in result['candidate_designs'])


@pytest.mark.parametrize('alpha', [0, .5, 1])
def test_zero_baseline_variance_and_alpha_endpoints(alpha):
    cfg = OGISInputs([.5]*4, [.5]*4, np.eye(4), [.5]*4, [.5]*4)
    d = LinkedDesign(cfg, {((0, 0), (1, 1)): .5, ((2, 2), (3, 3)): .5})
    result = optimize_gbfs(d, iterations=4, proposals=5, alpha=alpha)
    assert result['initial_score'] == 0
    assert np.isfinite(result['best_score']) and result['best_score'] == 0


def test_study_variables_do_not_affect_search():
    a = LinkedDesign.initialize(paper_inputs(y_A=[1, 2, 3, 4], y_B=[4, 3, 2, 1, 5]))
    b = LinkedDesign.initialize(paper_inputs(y_A=[9, 8, 7, 6], y_B=[1]*5))
    ra = optimize_gbfs(a, iterations=8, seed=3)
    rb = optimize_gbfs(b, iterations=8, seed=3)
    assert ra['history'] == rb['history']
    assert ra['best'].key() == rb['best'].key()


@pytest.mark.parametrize('kwargs', [dict(iterations=-1), dict(iterations=1.5),
    dict(proposals=0), dict(alpha=np.nan), dict(alpha=1.1), dict(fraction=0),
    dict(max_frontier=0), dict(keep_best=0), dict(stagnation_limit=0),
    dict(backtrack_depth=-1)])
def test_invalid_search_settings(kwargs):
    with pytest.raises(ValueError):
        optimize_gbfs(paper_design(), **kwargs)


def test_zero_iterations_returns_initial_and_complete_initial_history():
    d = paper_design()
    r = optimize_gbfs(d, iterations=0)
    assert r['best'] is d and r['generated_candidates'] == 1
    assert r['history'][0]['iteration'] == 0
    assert r['explored_states'] == 0


def test_zero_baseline_penalizes_later_positive_variance():
    # Initial strips balance x exactly; a legal exchange can unbalance them.
    cfg = OGISInputs([.5]*4, [.5]*4, np.eye(4), [1, 3, 2, 2], [.5]*4)
    d = LinkedDesign(cfg, {((0, 0), (1, 1)): .5, ((2, 2), (3, 3)): .5})
    r = optimize_gbfs(d, iterations=1, proposals=30, alpha=1, seed=12)
    assert r['initial_score'] == 0 and r['best_score'] == 0
    assert any(score > 0 for score, _ in r['candidate_designs'])
