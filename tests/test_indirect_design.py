"""Linked matchings must preserve the supplied design-based probabilities."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from graphical_sampling.indirect import OGISInputs, LinkedDesign, feasible_link_masses


def paper_inputs(**overrides):
    values = dict(pi_A=np.array([18, 12, 12, 6]) / 24,
                  pi_B=np.array([13, 15, 7, 10, 3]) / 24,
                  links=[[1, 1, 0, 0, 0], [1, 0, 1, 1, 0],
                         [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]],
                  x_A=[3, 1, 3, .5], x_B=np.ones(5))
    values.update(overrides)
    return OGISInputs(**values)


def pairs(*edges):
    return tuple(sorted((a - 1, b - 1) for a, b in edges))


def paper_design():
    return LinkedDesign(paper_inputs(), {
        pairs((1, 2), (2, 1)): 2/24, pairs((1, 1), (2, 3)): 2/24,
        pairs((1, 2), (2, 4)): 2/24, pairs((1, 1), (3, 2)): 6/24,
        pairs((1, 1), (3, 4)): 1/24, pairs((1, 2), (3, 4)): 5/24,
        pairs((2, 3), (4, 5)): 2/24, pairs((2, 1), (4, 5)): 1/24,
        pairs((2, 1), (4, 3)): 1/24, pairs((2, 4), (4, 3)): 2/24,
    })


def test_exact_manuscript_exchange():
    d = paper_design()
    assert d.nht_variance(d.cfg.x_A, d.cfg.x_B) == pytest.approx(
        (27/4, 1823/325), abs=1e-12)
    u, v = pairs((1, 1), (3, 2)), pairs((2, 3), (4, 5))
    after = d.exchange(u, v, (0, 0), (1, 2), fraction=.5)
    assert after is not None and after.validate()
    assert after.nht_variance(d.cfg.x_A, d.cfg.x_B) == pytest.approx(
        (73/12, 10841/2275), abs=1e-12)
    np.testing.assert_allclose(after.link_masses, d.link_masses, atol=1e-14, rtol=0)
    for design in [d, after]:
        sa, sb = design.sip()
        np.testing.assert_allclose(sa.diagonal(), d.cfg.pi_A, atol=1e-14, rtol=0)
        np.testing.assert_allclose(sb.diagonal(), d.cfg.pi_B, atol=1e-14, rtol=0)
        np.testing.assert_allclose(sa.sum(axis=1), d.cfg.n*d.cfg.pi_A)
        np.testing.assert_allclose(sb.sum(axis=1), d.cfg.n*d.cfg.pi_B)
        np.testing.assert_allclose(design.conditional_probabilities.sum(axis=1), 1)
    assert d.states[u] == 6/24  # exchanging must not modify the parent
    assert after.states[u] == pytest.approx(5/24)


@pytest.mark.parametrize('pi_a,pi_b', [([1, 1], [1, 1]),
    ([.5]*4, [.4]*5), ([1, .5, .5], [.5, .5, .5, .5]),
    ([.2, .8], [.4, .6]), ([1], [1])])
def test_initialization_recovers_marginals_and_lambda(pi_a, pi_b):
    c = OGISInputs(pi_a, pi_b, np.ones((len(pi_a), len(pi_b))),
                   np.ones(len(pi_a)), np.ones(len(pi_b)))
    lam = feasible_link_masses(c)
    d = LinkedDesign.initialize(c)
    assert d.validate()
    np.testing.assert_allclose(d.link_masses, lam, atol=1e-12, rtol=0)
    assert sum(d.states.values()) == pytest.approx(1)
    for edges in d.states:
        assert len({i for i, _ in edges}) == c.n
        assert len({k for _, k in edges}) == c.n


def test_infeasible_transport_even_when_every_unit_has_a_link():
    c = OGISInputs([1, 1], [.3, .3, .7, .7],
                   [[1, 1, 0, 0], [0, 0, 1, 1]], [1, 1], [1]*4)
    with pytest.raises(ValueError, match='[Ii]nfeasible'):
        LinkedDesign.initialize(c)


@pytest.mark.parametrize('overrides', [dict(pi_A=[0, .5, .5, 1]),
    dict(pi_A=[np.nan, .5, .5, .25]), dict(pi_A=[1.1, .4, .3, .2]),
    dict(pi_B=[.5]*5), dict(x_A=None), dict(x_A=[1, 2]),
    dict(y_B=[np.inf]*5), dict(links=np.full((4, 5), .5)),
    dict(links=np.full((4, 5), np.nan)), dict(links=np.zeros((4, 5))),
    dict(names_A=['a'])])
def test_invalid_inputs_are_rejected(overrides):
    with pytest.raises(ValueError):
        paper_inputs(**overrides)


@pytest.mark.parametrize('states', [
    {((0, 0), (0, 1)): 1}, {((0, 0), (1, 0)): 1},
    {((0, 2), (1, 0)): 1}, {((-1, 0), (1, 2)): 1},
    {((.1, 0), (1, 2)): 1}, {((0, 0),): 1},
    {((0, 0), (1, 2)): np.nan}, {((0, 0), (1, 2)): -1},
])
def test_invalid_strips_are_rejected(states):
    with pytest.raises(ValueError):
        LinkedDesign(paper_inputs(), states)


def test_forbidden_exchange_rejects_duplicates_without_repair():
    d = paper_design()
    # Replacing A1->B1 by A2->B3 would duplicate A2 in the first strip.
    u, v = pairs((1, 1), (2, 3)), pairs((2, 3), (4, 5))
    assert d.exchange(u, v, (0, 0), (1, 2)) is None
    # Distinct A endpoints can still duplicate B2.
    u, v = pairs((1, 1), (3, 2)), pairs((1, 2), (2, 4))
    assert d.exchange(u, v, (0, 0), (0, 1)) is None
    assert d.exchange(u, u, u[0], u[1]) is None
    with pytest.raises(ValueError):
        d.exchange(u, v, u[0], v[0], fraction=np.nan)


def test_input_and_search_snapshots_cannot_be_mutated():
    pi = np.array([.5, .5])
    c = OGISInputs(pi, pi, np.eye(2), [1, 2], [2, 3])
    pi[:] = 1
    np.testing.assert_array_equal(c.pi_A, [.5, .5])
    with pytest.raises((ValueError, FrozenInstanceError)):
        c.pi_A[0] = .25
    d = LinkedDesign(c, {((0, 0),): .5, ((1, 1),): .5})
    with pytest.raises(TypeError):
        d.states[((0, 0),)] = 1
    lm = d.link_masses
    lm[:] = 0
    assert d.validate()
    assert d.draw(seed=7) == d.draw(seed=7)


def test_tiny_positive_masses_and_key_probabilities_are_preserved():
    eps = 1e-12
    c = OGISInputs([eps, 1-eps], [eps, 1-eps], np.eye(2), [1, 2], [3, 4])
    d = LinkedDesign(c, {((0, 0),): eps, ((1, 1),): 1-eps})
    assert d.states[((0, 0),)] == eps
    assert d.key() != LinkedDesign(c, {((0, 0),): 2*eps,
                                      ((1, 1),): 1-2*eps}).key()


def test_fixed_lambda_and_nonfinite_evaluation_are_rejected():
    d = paper_design()
    with pytest.raises(ValueError):
        LinkedDesign(d.cfg, d.states, fixed_lambda=np.zeros((4, 5)))
    with pytest.raises(ValueError):
        d.nht_variance([np.nan]*4, [1]*5)
