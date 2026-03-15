from graphical_sampling.structs import Sample


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
