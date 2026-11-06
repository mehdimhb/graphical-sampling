"""Mathematical contracts for notebook 19, without running its display cells."""
from pathlib import Path
import nbformat
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'simulations/jupyters/19_exact_double_stratified_gfs.ipynb'

@pytest.fixture(scope='module')
def api():
    assert NOTEBOOK.exists(), 'The exact double-stratified notebook is missing'
    namespace = {}
    for cell in nbformat.read(NOTEBOOK, as_version=4).cells:
        if cell.cell_type == 'code' and 'definitions' in cell.metadata.get('tags', []):
            exec(compile(cell.source, str(NOTEBOOK), 'exec'), namespace)
    return namespace


def test_small_residual_construction_and_induced_design(api):
    pop = api['make_small_population']()
    X, labels, H = api['build_disjunctive_matrix'](pop)
    pi = pop.pi.to_numpy()
    m = api['check_integer_margins'](X, pi, H)
    np.testing.assert_array_equal(m, [2, 2, 2, 2])
    strips, trace = api['construct_exact_gfs'](X, pi, m, H)
    api['verify_gfs'](strips, X, pi, m, H)
    S, p, table = api['derive_design'](strips)
    np.testing.assert_allclose(p @ S, pi, atol=1e-10, rtol=0)
    assert len(trace) > 1
    assert np.all(p > 0)
    assert p.sum() == pytest.approx(1)


def test_flow_distinguishes_parallel_units_and_forces_boundaries(api):
    pop = api['make_small_population']()
    X, _, H = api['build_disjunctive_matrix'](pop)
    q = np.array([1, 0, .2, .8, .7, .3, 0, 1])
    s = api['find_exact_sample_flow'](X, q, np.array([2]*4), H)
    assert s[0] == s[7] == 1
    assert s[1] == s[6] == 0
    np.testing.assert_array_equal(X.T @ s, [2]*4)
    with pytest.raises(ValueError, match='infeasible'):
        api['find_exact_sample_flow'](X, np.zeros(8), np.array([2]*4), H)


@pytest.mark.parametrize('kind', ['same_cell', 'rectangle'])
def test_equal_height_split_exchange_preserves_lengths_and_margins(api, kind):
    pop = api['make_small_population']()
    X, _, H = api['build_disjunctive_matrix'](pop)
    if kind == 'same_cell':
        A = np.array([1,0,1,0,1,0,1,0])
        B = 1 - A
        move = api['find_same_cell_exchange'](A, B, X, H)
    else:
        A = np.array([1,1,0,0,0,0,1,1])
        B = 1 - A
        move = api['find_rectangle_exchange'](A, B, X, H)
    assert move is not None
    strips = [{'height': .4, 'sample': A}, {'height': .6, 'sample': B}]
    pi = api['reconstruct_pi'](strips)
    after, info = api['apply_exchange'](strips, 0, 1, .1, *move, X=X)
    assert len(after) == 4
    api['verify_gfs'](after, X, pi, np.array([2]*4), H)
    assert info['K'] == list(move[0])
    with pytest.raises(ValueError):
        api['apply_exchange'](strips, 0, 1, .5, *move, X=X)
    with pytest.raises(ValueError):
        api['apply_exchange'](strips, 0, 1, .1, [0], [2], X=X)


@pytest.mark.parametrize('value', [0., 1.])
def test_binary_terminal_population(api, value):
    pop = api['make_small_population']()
    X, _, H = api['build_disjunctive_matrix'](pop)
    pi = np.full(8, value)
    m = api['check_integer_margins'](X, pi, H)
    strips, _ = api['construct_exact_gfs'](X, pi, m, H)
    assert len(strips) == 1
    api['verify_gfs'](strips, X, pi, m, H)


def test_original_mu284_raking_and_same_construction(api):
    pop, source = api['load_mu284'](ROOT)
    assert len(pop) == 284
    pop, breaks = api['make_mu284_categories'](pop)
    np.testing.assert_array_equal(breaks, [3,10,16,31,653])
    X, _, H = api['build_disjunctive_matrix'](pop)
    assert X.shape == (284, 12) and H == 8
    pi0 = np.full(284, 100/284)
    expected = X.T @ pi0
    m = np.r_[api['round_margins'](expected[:H], 100),
              api['round_margins'](expected[H:], 100)]
    pi, history = api['rake_pi_to_integer_margins'](X, pi0, m, H)
    strips, _ = api['construct_exact_gfs'](X, pi, m, H)
    api['verify_gfs'](strips, X, pi, m, H)
    after, log = api['run_exchanges'](strips, X, pi, m, H, 20, seed=12345)
    assert len(log) == 20
    api['verify_gfs'](after, X, pi, m, H)
    assert {'same_cell', 'rectangle'} <= set(log.kind)


def test_missing_mu284_has_clear_local_data_instructions(api, tmp_path, monkeypatch):
    import sys
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api['shutil'], 'which', lambda name: None)
    monkeypatch.setitem(sys.modules, 'rpy2', None)
    with pytest.raises(FileNotFoundError, match='ORIGINAL 284-row MU284.csv'):
        api['load_mu284'](tmp_path)


def test_filtered_mu284_is_rejected(api, tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({'REG': np.ones(281), 'P85': np.ones(281)}).to_csv(tmp_path/'MU284.csv', index=False)
    with pytest.raises(ValueError, match='expected original MU284 with 284 rows, got 281'):
        api['load_mu284'](tmp_path)


def test_original_mu284_category_sizes_match_r_cut(api):
    pop, _ = api['load_mu284'](ROOT)
    pop, _ = api['make_mu284_categories'](pop)
    np.testing.assert_array_equal(pop.groupby('G').size(), [25,48,32,38,56,41,15,29])
    np.testing.assert_array_equal(pop.groupby('C').size(), [72,73,73,66])


def test_direct_random_length_stress_cases(api):
    result = api['run_stress_tests'](seed=602007, count=20)
    assert len(result) == 23
    assert result.passed.all()
    assert result.max_pi_error.max() <= 1e-10
    assert result.after_pi_error.max() <= 1e-10


def test_mu284_when_longdouble_has_only_float64_precision():
    namespace = {}
    for cell in nbformat.read(NOTEBOOK, as_version=4).cells:
        if cell.cell_type == 'code' and 'definitions' in cell.metadata.get('tags', []):
            source = cell.source.replace('np.longdouble', 'np.float64')
            exec(compile(source, '<float64-only notebook>', 'exec'), namespace)
    pop, _ = namespace['load_mu284'](ROOT)
    pop, _ = namespace['make_mu284_categories'](pop)
    X, _, H = namespace['build_disjunctive_matrix'](pop)
    pi0 = np.full(284, 100/284)
    expected = X.T @ pi0
    m = np.r_[namespace['round_margins'](expected[:H], 100),
              namespace['round_margins'](expected[H:], 100)]
    pi, _ = namespace['rake_pi_to_integer_margins'](X, pi0, m, H)
    strips, _ = namespace['construct_exact_gfs'](X, pi, m, H)
    namespace['verify_gfs'](strips, X, pi, m, H)
