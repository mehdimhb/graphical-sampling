from geometric_sampling.structs import Sample
from geometric_sampling.design import DesignGenetic
from geometric_sampling.GeneticOptimizer import GeneticOptimizer, EPSILON


def test_almost_zero():
    r = Sample(1e-10, frozenset({1, 2, 3}))
    assert r.almost_zero()


def test_compare():
    r1 = Sample(0.3, frozenset({1, 2}))
    r2 = Sample(0.5, frozenset({2, 3}))
    assert r1 < r2
    assert r2 > r1
    assert -r1 > -r2
    assert -r2 < -r1


def test_combine_fragments_n_with_epsilon():
    optimizer = GeneticOptimizer()

    frag1 = DesignGenetic()
    frag2 = DesignGenetic()

    frag1.push(Sample(EPSILON / 2, frozenset({1})))
    frag1.push(Sample(1.0, frozenset({1})))

    frag2.push(Sample(EPSILON / 2, frozenset({2})))
    frag2.push(Sample(1.0, frozenset({2})))

    combined = optimizer.combine_fragments_n([frag1, frag2])

    assert len(combined.heap) == 1
    only_sample = combined.pull()
    assert only_sample.probability == 1.0
    assert only_sample.ids == frozenset({1, 2})
