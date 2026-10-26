"""Exported distributions must be sufficient to reproduce reported results."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from graphical_sampling.indirect import (
    LinkedDesign, optimize_gbfs, export_results, plot_results,
)
from test_indirect_design import paper_design


def test_export_round_trip_keeps_every_discovered_distribution(tmp_path):
    r = optimize_gbfs(paper_design(), iterations=5, proposals=12, seed=6)
    files = export_results(r, tmp_path)
    assert all(Path(p).is_file() for p in files)
    with gzip.open(tmp_path/'all_discovered_candidate_designs.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == r['generated_candidates']
    scores = pd.read_csv(tmp_path/'all_discovered_candidate_scores.csv')
    assert len(scores) == len(records)
    for rec, (score, design) in zip(records, r['candidate_designs']):
        rebuilt = LinkedDesign(design.cfg, {
            tuple(map(tuple, row['edges'])): row['probability'] for row in rec['states']
        }, fixed_lambda=design.fixed_lambda)
        assert rebuilt.key() == design.key()
        assert rec['score'] == score
    for name in ['initial_all_sample_pairs.csv', 'optimized_all_sample_pairs.csv']:
        assert pd.read_csv(tmp_path/name).probability.sum() == pytest.approx(1, abs=1e-14)
    np.testing.assert_allclose(pd.read_csv(tmp_path/'conditional_probabilities.csv', index_col=0).sum(axis=1), 1)
    metadata = json.loads((tmp_path/'run_metadata.json').read_text())
    assert metadata['seed'] == 6 and metadata['heuristic'] is True
    assert metadata['settings']['iterations'] == 5
    assert metadata['settings']['proposals'] == 12
    assert (tmp_path/'ogis_figures.png').stat().st_size > 1000
    fig = plot_results(r)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_export_evaluates_independently_optional_study_variables(tmp_path):
    from test_indirect_design import paper_inputs
    d = LinkedDesign.initialize(paper_inputs(y_A=[5, 2, 6, 1]))
    r = optimize_gbfs(d, iterations=0, seed=np.int64(3))
    export_results(r, tmp_path)
    comparisons = pd.read_csv(tmp_path/'variance_comparison.csv')
    study = comparisons[comparisons.variable == 'study']
    assert study.population.tolist() == ['A']
    assert np.isfinite(study.before.iloc[0])


def test_notebook_run_all_from_input_cell_exports_complete_results(tmp_path, monkeypatch):
    notebook = Path(__file__).resolve().parents[1]/'simulations/jupyters/20_ogis_indirect_sampling.ipynb'
    assert notebook.exists(), 'The package-backed OGIS notebook is missing'
    cells = json.loads(notebook.read_text())['cells']
    assert sum('parameters' in c.get('metadata', {}).get('tags', []) for c in cells) == 1
    monkeypatch.chdir(tmp_path)
    namespace = {'__name__': '__main__'}
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
        exec(compile(''.join(cell['source']), f'ogis-notebook-cell-{i}', 'exec'), namespace)
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            namespace.update(ITERATIONS=3, PROPOSALS_PER_STEP=10,
                             OUTPUT_DIR=str(tmp_path/'exports'))
    result = namespace['result']
    assert result['best'].validate()
    assert (tmp_path/'exports/all_discovered_candidate_designs.jsonl.gz').exists()
    assert len(namespace['candidate_scores']) == result['generated_candidates']
    plt.close('all')
