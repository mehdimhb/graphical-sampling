"""Global GFS invariants; fractional cluster membership is a separate property."""

from collections import defaultdict
from itertools import product
import math
from pathlib import Path
import pickle

import numpy as np
import pytest

from graphical_sampling import Design, Order, Population
from graphical_sampling.clustering.fip_balanced_nmeans import FIPBalancedNMeans
from graphical_sampling.order import Ceil, Cluster, Floor, Zone
from graphical_sampling.search.gbfs import GreedyBestFirstSearch


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = [
    "best_design_df_meuse_30_pik_ep.pkl", "best_design_df_meuse_3_pik_up.pkl",
    "initial_design_df_meuse_3_pik_up.pkl", "initial_design_df_swiss_20_pik_ep.pkl",
]
SAVED = sorted((ROOT / "simulations/best_designs").rglob("*.pkl")) + [
    ROOT / "simulations/Store/best_design" / name for name in ARCHIVED
]


def raw_law(design):
    masses = defaultdict(list)
    for heap in design._heaps:
        for sample in heap:
            assert np.isfinite(sample.prob) and sample.prob > 0
            masses[tuple(sorted(sample.ids))].append(float(sample.prob))
    return {s: math.fsum(p) for s, p in masses.items()}


def assert_valid(design, tolerance=3e-7):
    # Inspect raw heaps before the API can merge/normalize their probabilities.
    law = raw_law(design)
    assert law and math.fsum(law.values()) == pytest.approx(1, abs=tolerance)
    fip = np.zeros(design.pop.N)
    for sample, prob in law.items():
        assert len(sample) == len(set(sample)) == design.pop.n
        assert all(0 <= i < design.pop.N for i in sample)
        fip[list(sample)] += prob
    np.testing.assert_allclose(fip, design.pop.inclusions, atol=tolerance, rtol=0)
    samples, probs = design.all_samples_and_probs
    assert samples.shape[1] == design.pop.n
    assert np.all(np.isfinite(probs)) and np.all(probs > 0)
    assert math.fsum(map(float, probs)) == pytest.approx(1, abs=tolerance)
    api = defaultdict(float)
    for sample, prob in zip(samples, probs):
        assert len(set(sample)) == design.pop.n
        api[tuple(sorted(sample))] += float(prob)
    assert api.keys() == law.keys()
    for sample in law:
        assert api[sample] == pytest.approx(law[sample], abs=tolerance)
    return law


def systematic_reference(pop, permutation):
    """Independent geometric oracle: place U+k on the full cumulative mass line."""
    ends = np.cumsum(pop.inclusions[permutation])
    breaks = np.unique(np.r_[0., np.remainder(ends, 1.), 1.])
    law = defaultdict(float)
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right - left < 1e-10:
            continue
        points = (left + right) / 2 + np.arange(pop.n)
        selected = permutation[np.searchsorted(ends, points, side="right")]
        assert len(np.unique(selected)) == pop.n
        law[tuple(sorted(selected))] += right - left
    return law


def assert_reference_law(design):
    ids = design.order.get()[:, 0].astype(int)
    _, first = np.unique(ids, return_index=True)
    permutation = ids[np.sort(first)]
    assert sorted(permutation) == list(range(design.pop.N))
    shares = np.bincount(ids, weights=design.order.get()[:, 1], minlength=design.pop.N)
    np.testing.assert_allclose(shares, 1, atol=2e-12, rtol=0)
    actual = assert_valid(design)
    expected = systematic_reference(design.pop, permutation)
    assert actual.keys() == expected.keys()
    for sample in actual:
        assert actual[sample] == pytest.approx(expected[sample], abs=2e-8)


@pytest.mark.parametrize("n,q", [(5, 4), (10, 2), (20, 4)])
def test_split_orders_keep_n_distinct_units_with_prescribed_ips(n, q):
    N = 3 * n + 1
    pop = Population(np.random.default_rng(n).random((N, 2)), np.ones(N), n=n)
    zone = Zone(np.ones(N), np.arange(N), list(range(N)))
    order = Order.from_clusters(pop, [Cluster(0, [zone])], num_splits=q,
                                topological=False)
    assert_reference_law(Design.from_order(pop, order))


@pytest.fixture(scope="module", params=list(product(
    [5, 10, 20], ["EP", "UP"], [0, 17], ["cluster", "sweep_xy"]
)))
def fitted_order(request):
    n, ip, seed, mode = request.param
    rng = np.random.default_rng(seed)
    N = 6 * n + 1
    weights = np.ones(N) if ip == "EP" else rng.uniform(.5, 1.5, N)
    pop = Population(rng.random((N, 2)), weights, n=n)
    solver = FIPBalancedNMeans(n, init_clust_method="weighted", n_init=1,
                              random_state=seed)
    solver.fit(pop)
    original_membership = solver.membership.copy()
    assert np.any((original_membership > 0) & (original_membership < 1))
    np.testing.assert_allclose(original_membership.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(pop.inclusions @ original_membership, 1, atol=1e-10)
    solver.fit_zones(2 if mode == "cluster" else (2, 2), mode=mode)
    assert all(c.floor.index == c.ceil.index == -1 for c in solver.clusters)
    after = solver._generate_membership(solver.clusters, N, pop.indices)
    np.testing.assert_allclose(after, original_membership, atol=1e-10)
    for c in solver.clusters:
        for z in c.zones:
            assert pop.inclusions[z.indices] @ z.shares == pytest.approx(
                1 / len(c.zones), abs=1e-10)
    return Order.from_clusters(pop, solver.clusters, topological=False), seed


@pytest.mark.parametrize("q", [1, 2, 4])
def test_fractional_zones_reordering_flattening_and_exchanges_preserve_ips(fitted_order, q):
    template, seed = fitted_order
    order = template.copy()
    order.num_splits = q
    order._build_order(q)
    design = Design.from_order(order.pop, order)
    assert_reference_law(design)

    # Exercise actual neighbor operators without running an optimization/search.
    for changes, zone_changes in [(2, 0), (0, 2), (2, 2)]:
        changed = GreedyBestFirstSearch._apply_order_change(
            design, order.pop.n, order.num_zones, changes, zone_changes, None)
        assert_reference_law(changed)
    flat = design.flatten()
    assert len(flat.order.clusters) == len(flat.order.clusters[0].zones) == 1
    assert_reference_law(flat)
    global_changed = GreedyBestFirstSearch._apply_order_change(flat, 1, 1, 8, 0, 3)
    assert_reference_law(global_changed)
    global_changed._rng = np.random.default_rng(seed)
    for _ in range(5):
        global_changed.exchange(partitions=1, exchange_coef=.5)
    after = assert_valid(global_changed)
    draws = global_changed.sample(32)
    assert draws.shape == (32, order.pop.n)
    assert all(tuple(sorted(s)) in after for s in draws)


@pytest.mark.parametrize("seed", [0, 1, 17])
def test_nontrivial_probability_mass_exchange_preserves_size_and_ips(seed):
    pop = Population(np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]),
                     np.full(4, .5))
    zone = Zone(np.ones(4), np.arange(4), list(range(4)))
    order = Order.from_clusters(pop, [Cluster(0, [zone])], topological=False)
    design = Design.from_order(pop, order)
    assert raw_law(design) == {(0, 2): .5, (1, 3): .5}
    design._rng = np.random.default_rng(seed)
    design.exchange(partitions=1, exchange_coef=.5)
    law = assert_valid(design)
    assert len(law) == 4
    assert list(law.values()) == [.25] * 4


def test_collapse_preserves_ips_but_not_weighted_or_argmax_cluster_counts():
    pop = Population(np.array([[0., 0.], [1., 0.], [2., 0.]]), np.array([.6, .6, .8]))
    solver = FIPBalancedNMeans(2)
    solver.pop = pop
    solver.clusters = [
        Cluster(0, [Zone(np.ones(1), np.array([0]), [0])],
                Floor(-1, 0.), Ceil(1, 2 / 3)),
        Cluster(1, [Zone(np.ones(1), np.array([2]), [0])],
                Floor(1, 1 / 3), Ceil(-1, 0.)),
    ]
    membership = np.array([[1., 0.], [2 / 3, 1 / 3], [0., 1.]])
    np.testing.assert_allclose(pop.inclusions @ membership, [1., 1.])
    solver.fit_zones((1, 1), mode="sweep_xy")
    z = solver.clusters[0].zones[0]
    z.sort = [int(np.flatnonzero(z._indices == i)[0]) for i in [1, 0]]
    order = Order.from_clusters(pop, solver.clusters, topological=False)
    assert not order.fixed_ids
    design = Design.from_order(pop, order)
    law = assert_valid(design)
    assert law[(0, 1)] == pytest.approx(.2)
    # Both selected units have their largest original membership in cluster 0.
    assert np.argmax(membership[[0, 1]], axis=1).tolist() == [0, 0]
    np.testing.assert_allclose(membership[[0, 1]].sum(axis=0), [5 / 3, 1 / 3])
    # Global size/IP correctness is independent of this stronger cluster claim.
    assert_reference_law(design)


def test_fractional_border_reordering_can_leave_an_original_cluster_unrepresented():
    pop = Population(np.column_stack([np.arange(4.), np.zeros(4)]),
                     np.array([.4, .4, .4, .8]))
    solver = FIPBalancedNMeans(2)
    solver.pop = pop
    solver.clusters = [
        Cluster(0, [Zone(np.ones(2), np.array([0, 1]), [0, 1])],
                Floor(-1, 0.), Ceil(2, .5)),
        Cluster(1, [Zone(np.ones(1), np.array([3]), [0])],
                Floor(2, .5), Ceil(-1, 0.)),
    ]
    membership = np.array([[1., 0.], [1., 0.], [.5, .5], [0., 1.]])
    np.testing.assert_allclose(pop.inclusions @ membership, [1., 1.])
    solver.fit_zones((1, 1), mode="sweep_xy")
    z = solver.clusters[0].zones[0]
    z.sort = [int(np.flatnonzero(z._indices == i)[0]) for i in [0, 2, 1]]
    design = Design.from_order(pop, Order.from_clusters(pop, solver.clusters,
                                                     topological=False))
    law = assert_valid(design)
    assert law[(0, 1)] == pytest.approx(.2)
    # Neither selected unit has even a fractional membership in cluster 1;
    # unlike argmax/weighted counts, this rules out any one-per-cluster assignment.
    assert not np.any(membership[[0, 1], 1])


def test_flattened_reordering_can_select_two_units_from_one_original_cluster():
    # No border exists to protect: global reordering alone loses this property.
    pop = Population(np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]),
                     np.full(4, .5))
    clusters = [Cluster(j, [Zone(np.ones(2), np.array(ids), [0, 1])])
                for j, ids in enumerate([[0, 1], [2, 3]])]
    original = Design.from_order(pop, Order.from_clusters(pop, clusters, topological=False))
    assert raw_law(original) == {(0, 2): .5, (1, 3): .5}
    flat = original.flatten()
    flat.order.clusters[0].zones[0].sort = [0, 2, 1, 3]
    flat.order._build_order(1)
    reordered = Design.from_order(pop, flat.order)
    assert_valid(reordered)
    assert raw_law(reordered) == {(0, 1): .5, (2, 3): .5}


@pytest.mark.parametrize("path", SAVED, ids=lambda p: p.name)
def test_saved_nms_gms_designs_retain_complete_unit_masses_and_valid_laws(path):
    # Read only: no resaving, normalization on disk, or optimization.
    with path.open("rb") as stream:
        design = pickle.load(stream)
    assert_valid(design)
    order = design.order.get()
    ids = order[:, 0].astype(int)
    assert set(ids) == set(range(design.pop.N))
    shares = np.bincount(ids, weights=order[:, 1], minlength=design.pop.N)
    np.testing.assert_allclose(shares, 1, atol=2e-12, rtol=0)
    # Historical fragment counts need not equal N, but their contiguous pieces
    # can be coalesced to exactly one full bar for every original unit.
    runs = ids[np.r_[True, ids[1:] != ids[:-1]]]
    assert len(runs) == len(np.unique(runs)) == design.pop.N
    # Replaying the retained order is valid; exchanges may make its law differ
    # from the retained final probability heaps, so do not require equality.
    assert_valid(Design.from_order(design.pop, design.order.copy()))
