"""Apply the supplied manuscript's spatial-panel style to saved results only."""
import os

os.environ['MPLCONFIGDIR'] = '/tmp/jssam-paper-style-mpl'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_format'
OUT.mkdir(exist_ok=True)
cfg = json.loads((ROOT / 'config.json').read_text())
comparisons = json.loads((ROOT / 'comparisons.json').read_text())
assert json.loads((ROOT / 'verification.json').read_text())['all_checks_passed']
xy = np.array([(x, y) for y in range(12) for x in range(12)]) / 11
palette = plt.get_cmap('tab10')(np.arange(8))
POPULATION_AREA = 12  # Larger dots expose the existing UP size variation.


def load(run_id):
    return json.loads((ROOT / 'runs' / run_id / 'summary.json').read_text())


def align(a, b):
    a, b = np.asarray(a['labels']), np.asarray(b['labels'])
    overlap = np.array([[np.sum((a == i) & (b == j)) for j in range(8)]
                        for i in range(8)])
    i, j = linear_sum_assignment(-overlap)
    mapping = np.empty(8, dtype=int)
    mapping[j] = i
    return mapping


def four(value):
    # Avoid displaying negative zero after rounding.
    return f'{0.0 if abs(value) < 0.00005 else value:.4f}'


plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.grid': False,
})
fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.5))
panel_records = []
for row_index, row in enumerate(cfg['rows']):
    states = [load(f"{row['id']}_{kind}_{where}")
              for kind in ['equal', 'unequal']
              for where in ['original', 'shifted']]
    maps = [np.arange(8), align(states[0], states[1]), align(states[0], states[2])]
    maps.append(maps[2][align(states[2], states[3])])
    for col, (ax, state, mapping) in enumerate(zip(axes[row_index], states, maps)):
        labels = np.asarray(state['labels'])
        probabilities = np.asarray(state['pi'])
        for cluster in range(8):
            mask = labels == cluster
            points = xy[mask]
            color = palette[mapping[cluster]]
            if len(points) >= 3 and np.linalg.matrix_rank(points - points[0]) == 2:
                hull = ConvexHull(points)
                ax.fill(*points[hull.vertices].T, facecolor=color,
                        edgecolor='none', alpha=.10, zorder=0)
            ax.scatter(*points.T, s=POPULATION_AREA * probabilities[mask] / (8 / 144),
                       color=color, alpha=.85, linewidths=0, zorder=1)
        medoids = xy[state['medoid_ids']]
        matched = xy[state['matched_ids']]
        for medoid, sample_unit in zip(medoids, matched):
            ax.plot([medoid[0], sample_unit[0]], [medoid[1], sample_unit[1]],
                    color='black', lw=.7, ls='--', zorder=2)
        sample = np.asarray(state['sample']) / 11
        ax.scatter(*sample.T, s=25, c='#E64B35', edgecolors='black',
                   linewidths=.5, zorder=3)
        ax.scatter(*medoids.T, s=34, c='black', marker='x',
                   linewidths=1.2, zorder=4)
        letter = chr(97 + 4 * row_index + col)
        kind = 'EP' if col < 2 else 'UP'
        position = 'shifted' if col % 2 else 'original'
        description = f'Scenario {row_index + 1}'
        stats = f"DI = {four(state['DI'])}"
        delta = None
        if col % 2:
            comparison = next(c for c in comparisons
                              if c['row'] == row['id'] and c['FIP'] == state['kind'])
            delta = comparison['DI_difference']
            stats += rf"; $\Delta$DI = {four(delta)}"
        ax.set_title(f'({letter}) {description}: {kind}, {position}\n{stats}',
                     pad=6, linespacing=1.15)
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), aspect='equal',
               xlabel=r'$c_1$', ylabel=r'$c_2$')
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.labelpad = 10
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(length=3, width=.6, pad=2)
        panel_records.append({'panel': letter, 'run_id': state['run_id'],
                              'scenario': row_index + 1, 'probabilities': kind,
                              'DI': state['DI'], 'eta': state['eta'],
                              'DI_difference': delta,
                              'displayed_difference': None if delta is None else four(delta)})

legend = [
    Line2D([], [], marker='o', ls='', mfc='#E64B35', mec='black', mew=.5,
           ms=5, label='sampled unit'),
    Line2D([], [], marker='x', ls='', color='black', mew=1.2,
           ms=6, label='IPnM medoid'),
    Line2D([], [], color='black', ls='--', lw=.7,
           label='medoid → matched unit'),
    Line2D([], [], marker='o', ls='', color='tab:blue', mec='none',
           ms=np.sqrt(POPULATION_AREA), label=r'population unit (area $\propto\pi_i$)'),
]
fig.subplots_adjust(left=.046, right=.993, bottom=.10, top=.93,
                    wspace=.25, hspace=.39)
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .017),
           ncol=4, frameon=False, fontsize=9, columnspacing=2)
stem = OUT / 'equal_unequal_translation_paper'
fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.05)
fig.savefig(stem.with_suffix('.png'), dpi=400, bbox_inches='tight', pad_inches=.05)
plt.close(fig)
(OUT / 'panels.json').write_text(json.dumps(panel_records, indent=2) + '\n')
(OUT / 'format_notes.txt').write_text(
    'Paper-style version of the eight verified saved spatial panels.\n'
    'EP = equal inclusion probabilities; UP = unequal inclusion probabilities.\n'
    'Rows are Scenario 1 and Scenario 2, respectively.\n'
    'Coordinates displayed as c1=x/11, c2=y/11. Cluster colors aligned by overlap.\n'
    'Pale polygons are convex hulls of cluster members, as in the supplied reference;\n'
    'colored population points give the actual memberships. Point area is proportional\n'
    'to the population unit inclusion probability on the same scale in every panel.\n'
    'Population markers enlarged to make unequal inclusion probabilities more visible;\n'
    'area = 12*pi_i/(8/144) square points. Eta is omitted from panel titles.\n'
    'Red circles: sampled units. Black crosses: medoids. Dashed lines: saved matching.\n'
    'Top row shift: (1,1) raw grid steps; bottom row shift: (9,8).\n'
    'Values displayed to four decimal places; full precision retained in panels.json\n'
    'and the original saved results. No fits or scientific calculations rerun.\n'
)
(OUT / 'caption.txt').write_text(
    'Sample translation under equal (EP) and unequal (UP) inclusion probabilities '
    'for a fixed 12 × 12 population (N = 144, n = 8). '
    'The top and bottom rows show Scenario 1 and Scenario 2, with sample shifts '
    'of (1, 1) and (9, 8) grid steps, respectively. Columns show the original and '
    'shifted samples under EP, followed by the original and shifted samples under UP. '
    'Coordinates are normalized to [0, 1]. Colored points identify population units '
    'and their cluster memberships; pale polygons show cluster convex hulls. '
    'Population-point areas are proportional to inclusion probabilities, which '
    'remain attached to the fixed population units. Red circles mark sampled units, '
    'black crosses mark IPnM medoids, and dashed lines show the medoid-to-sample '
    'matching. Each panel uses an independently fitted reference. '
    'ΔDI = DI_shifted − DI_original denotes the change in DI within each '
    'scenario and probability setting. Values are displayed to four decimal places.\n'
)
(OUT / 'caption.tex').write_text(
    r'\caption{Sample translation under equal (EP) and unequal (UP) inclusion '
    r'probabilities for a fixed $12\times12$ population ($N=144$, $n=8$). '
    r'The top and bottom rows show Scenario 1 and Scenario 2, with sample shifts '
    r'of $(1,1)$ and $(9,8)$ grid steps, respectively. Columns show the original '
    r'and shifted samples under EP, followed by the original and shifted samples '
    r'under UP. Coordinates are normalized to $[0,1]$. Colored points identify '
    r'population units and their cluster memberships; pale polygons show cluster '
    r'convex hulls. Population-point areas are proportional to inclusion '
    r'probabilities, which remain attached to the fixed population units. '
    r'Red circles mark sampled units, black crosses mark IPnM medoids, and dashed '
    r'lines show the medoid-to-sample matching. Each panel uses an independently '
    r'fitted reference. $\Delta\mathrm{DI}=\mathrm{DI}_{\mathrm{shifted}}'
    r'-\mathrm{DI}_{\mathrm{original}}$ denotes the change in DI within each '
    r'scenario and probability setting. Values are displayed to four decimal places.}'
    '\n'
)
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(OUT.iterdir())]
            if p.is_file() and p.name != 'manifest.json'}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(stem.with_suffix('.pdf'))
print(stem.with_suffix('.png'))
