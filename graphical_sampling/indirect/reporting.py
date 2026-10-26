"""Plots and complete research exports for linked sampling designs."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def variance_comparison(result):
    """Auxiliary and available simulation-only study NHT total variances."""
    cfg = result['initial'].cfg
    rows = []
    for kind in ('x', 'y'):
        a, b = getattr(cfg, f'{kind}_A'), getattr(cfg, f'{kind}_B')
        if a is None and b is None:
            continue
        values = (np.zeros_like(cfg.pi_A) if a is None else a,
                  np.zeros_like(cfg.pi_B) if b is None else b)
        before = result['initial'].nht_variance(*values)
        after = result['best'].nht_variance(*values)
        for j, (pop, available) in enumerate([('A', a), ('B', b)]):
            if available is not None:
                rows.append(dict(variable='auxiliary' if kind == 'x' else 'study',
                                 population=pop, before=before[j], after=after[j],
                                 improvement_percent=100*(1-after[j]/before[j]) if before[j] else np.nan))
    return pd.DataFrame(rows)


def candidate_table(result):
    """All unique discovered candidates in discovery order, including initial."""
    rows = []
    for j, ((score, design), iteration) in enumerate(zip(
            result['candidate_designs'], result['candidate_iterations'])):
        va, vb = design.nht_variance(design.cfg.x_A, design.cfg.x_B)
        rows.append(dict(candidate_number=j+1, iteration=iteration, score=score,
                         supported_matchings=len(design.states),
                         auxiliary_variance_A=va, auxiliary_variance_B=vb))
    return pd.DataFrame(rows)


def plot_links(cfg, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ya, yb = np.linspace(1, 0, len(cfg.pi_A)), np.linspace(1, 0, len(cfg.pi_B))
    for i, k in zip(*np.where(cfg.links)):
        ax.plot([0, 1], [ya[i], yb[k]], color='.75', lw=1.4, zorder=0)
    ax.scatter(np.zeros(len(ya)), ya, s=100+250*cfg.pi_A, color='#3569a8', zorder=2)
    ax.scatter(np.ones(len(yb)), yb, s=100+250*cfg.pi_B, color='#379575', zorder=2)
    for pos, yy, names, probabilities, align in [
            (-.06, ya, cfg.names_A, cfg.pi_A, 'right'),
            (1.06, yb, cfg.names_B, cfg.pi_B, 'left')]:
        for y, label, p in zip(yy, names, probabilities):
            ax.text(pos, y, f'{label}  ({p:.3f})', ha=align, va='center', fontsize=9)
    ax.set(xlim=(-.6, 1.6), ylim=(-.1, 1.1), title='Admissible links · marker area reflects FIP')
    ax.axis('off')
    return ax


def plot_design(design, ax=None):
    """Show every supported strip; each colored piece is an intact A->B pair."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    colors = plt.get_cmap('tab20')
    position = 0.
    for edges, p in design.states.items():
        for i, k in edges:
            ax.add_patch(Rectangle((i-.4, position), .8, p,
                                   facecolor=colors(k % 20), edgecolor='white', lw=.5))
            if p > .035:
                ax.text(i, position+p/2, design.cfg.names_B[k], ha='center', va='center', fontsize=7)
        position += p
    ax.set(xlim=(-.6, len(design.cfg.pi_A)-.4), ylim=(0, 1),
           xticks=range(len(design.cfg.pi_A)), xticklabels=design.cfg.names_A,
           xlabel='Frame A · nested colors identify assigned B units',
           ylabel='Strip probability axis',
           title=f'All {len(design.states)} supported linked matchings')
    ax.legend(handles=[Patch(color=colors(k % 20), label=name)
                       for k, name in enumerate(design.cfg.names_B)],
              fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    return ax


def plot_results(result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    plot_links(result['initial'].cfg, axes[0, 0])
    plot_design(result['initial'], axes[0, 1])
    axes[0, 1].set_title(f'Initial · all {len(result["initial"].states)} supported matchings')
    plot_design(result['best'], axes[1, 0])
    axes[1, 0].set_title(f'Optimized · all {len(result["best"].states)} supported matchings')
    history = result['history']
    axes[1, 1].plot([r['iteration'] for r in history], [r['best_score'] for r in history],
                    color='#3569a8', lw=2)
    axes[1, 1].set(xlabel='GBFS iteration', ylabel='Best normalized auxiliary variance',
                   title='Package GBFS reservoir progress')
    return fig


def export_results(result, directory):
    """Export support, SIPs, lambda, conditional assignments and full history.

    gzip JSONL retains every unique discovered candidate's complete numerical
    distribution, using zero-based pair indices. CSV labels are for display.
    Existing files with these names in ``directory`` are overwritten.
    """
    import matplotlib.pyplot as plt

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    cfg = result['initial'].cfg
    tables = {
        'initial_all_sample_pairs.csv': pd.DataFrame(result['initial'].sample_table()),
        'optimized_all_sample_pairs.csv': pd.DataFrame(result['best'].sample_table()),
        'variance_comparison.csv': variance_comparison(result),
        'gbfs_progress.csv': pd.DataFrame(result['history']),
        'all_discovered_candidate_scores.csv': candidate_table(result),
    }
    for name, frame in tables.items():
        frame.to_csv(out/name, index=False)
    for prefix, design in [('initial', result['initial']), ('optimized', result['best'])]:
        sa, sb = design.sip()
        for suffix, matrix, names in [('A', sa, cfg.names_A), ('B', sb, cfg.names_B)]:
            pd.DataFrame(matrix, index=names, columns=names).to_csv(out/f'{prefix}_SIP_{suffix}.csv')
    for name, matrix in [('link_masses', result['best'].link_masses),
                          ('conditional_probabilities', result['best'].conditional_probabilities)]:
        pd.DataFrame(matrix, index=cfg.names_A, columns=cfg.names_B).to_csv(out/f'{name}.csv')
    metadata = {name: (None if getattr(cfg, name) is None else np.asarray(getattr(cfg, name)).tolist())
                for name in ['pi_A', 'pi_B', 'links', 'x_A', 'x_B', 'y_A', 'y_B', 'names_A', 'names_B']}
    metadata.update(n=cfg.n, alpha=result['alpha'], scales=result['scales'], seed=result['seed'],
                    settings=result['settings'],
                    heuristic=True, index_base=0, generated_candidates=result['generated_candidates'],
                    initial_score=result['initial_score'], best_score=result['best_score'])
    (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    with gzip.open(out/'all_discovered_candidate_designs.jsonl.gz', 'wt', encoding='utf8') as stream:
        for j, ((score, design), iteration) in enumerate(zip(
                result['candidate_designs'], result['candidate_iterations'])):
            rec = dict(candidate_number=j+1, iteration=iteration, score=score,
                       states=[dict(edges=edges, probability=p) for edges, p in design.states.items()])
            stream.write(json.dumps(rec, allow_nan=False)+'\n')
    fig = plot_results(result)
    try:
        fig.savefig(out/'ogis_figures.png', dpi=150)
        fig.savefig(out/'ogis_figures.pdf')
    finally:
        plt.close(fig)
    names = [*tables, 'initial_SIP_A.csv', 'initial_SIP_B.csv',
             'optimized_SIP_A.csv', 'optimized_SIP_B.csv', 'link_masses.csv',
             'conditional_probabilities.csv', 'run_metadata.json',
             'all_discovered_candidate_designs.jsonl.gz', 'ogis_figures.png', 'ogis_figures.pdf']
    return [out/name for name in names]
