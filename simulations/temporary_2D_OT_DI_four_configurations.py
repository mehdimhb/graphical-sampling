"""Publication-style 2x3 illustration of translation and rotation DI (n=4)."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.index import DensityDisparity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulations/results/temporary_2D_relative_rotation/2D_DI_six_sample_configurations_n2_column_groups.png"
OUT_PDF = OUT.with_suffix(".pdf")

# Regular-square population used by the sample-adaptive reference illustration.
xx, yy = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 20))
coords = np.column_stack([xx.ravel(), yy.ravel()])
N, n = len(coords), 2
pik = np.full(N, n / N)
population = SimpleNamespace(
    coords=np.ascontiguousarray(coords), inclusions=pik, indices=np.arange(N),
    ids=np.arange(N), variable=np.zeros(N), N=N, n=n,
)
scorer = DensityDisparity(population, representative="nmedoids", n_jobs=1)


class SampleInitializedOT(FIPBalancedNMeans):
    def _get_labels_centroids(self, x, p, init_centroids=None):
        supplied = np.asarray(init_centroids, dtype=float)
        assert np.array_equal(supplied, self._supplied)
        return self._get_labels_centroids_ot_kmeans(x, p, supplied.copy())

    def fit_sample(self, pop, sample):
        self._supplied = np.asarray(sample, dtype=float).copy()
        return self.fit(pop, init_centroids=self._supplied)


def fit_ot(sample):
    # _get_labels_centroids_ot_kmeans constructs Sinkhorn with epsilon=0.01.
    fit = SampleInitializedOT(n=n, init_clust_method="ot")
    fit.fit_sample(population, sample)
    return fit


def get_reference():
    # Aligned initial centers yield two vertical balanced regions. Rotating this
    # medoid pattern creates genuinely different OT attraction basins.
    centers = np.column_stack([np.linspace(.20, .80, n), np.full(n, .5)])
    for _ in range(12):
        fit = fit_ot(centers)
        medoids, _ = scorer._cluster_medoids(fit.labels, fit.centroids)
        if np.allclose(centers, medoids):
            break
        centers = medoids
    return medoids


def transform(refs, angle, scale, shift):
    pivot = refs.mean(axis=0)
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    targets = pivot + scale*(refs-pivot)@R.T + shift
    # A realized sample must consist of population units. Select distinct nearest
    # grid units to the ideal transformed locations by one-to-one assignment.
    target_rows, population_cols = linear_sum_assignment(cdist(targets, coords))
    sample = np.empty_like(targets)
    sample[target_rows] = coords[population_cols]
    return sample


def evaluate(sample):
    fit = fit_ot(sample)
    medoids, _ = scorer._cluster_medoids(fit.labels, fit.centroids)
    rows, cols = linear_sum_assignment(cdist(medoids, sample))
    assigned = np.empty_like(medoids); assigned[rows] = sample[cols]
    translated = coords + (assigned-medoids)[fit.labels]
    di = scorer._score_single_density(scorer._density(translated))
    return fit.labels, medoids, assigned, float(di)


refs = get_reference()
# Two exact grid increments right and one grid increment down.
shift_1 = np.array([2/29, -1/19])
shift_2 = np.array([-2/29, 2/19])
specs = [
    # Row-major order gives one transformation family per figure column.
    ("(a) Shifted sample", 0, shift_1, "Moved right and down"),
    ("(c) Rotated sample", 45, np.zeros(2), r"Rotated $45^\circ$ counterclockwise"),
    ("(e) Rotated and shifted sample", 45, shift_1,
     r"Rotated $45^\circ$, then moved right and down"),
    ("(b) Shifted sample", 0, shift_2, "Moved left and up"),
    ("(d) Rotated sample", 75, np.zeros(2), r"Rotated $75^\circ$ counterclockwise"),
    ("(f) Rotated and shifted sample", 75, shift_2,
     r"Rotated $75^\circ$, then moved left and up"),
]
colors = ["#168BD2", "#FF7A00", "#24A148", "#9C4DCC"]

fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.6), sharex=True, sharey=True)
outcomes = []
all_labels = []
for ax, (heading, angle, shift, description) in zip(axes.flat, specs):
    sample = transform(refs, angle, 1.0, shift)
    labels, medoids, assigned, di = evaluate(sample)
    outcomes.append((heading, angle, shift.copy(), di))
    all_labels.append(labels.copy())
    for k, color in enumerate(colors):
        mask = labels == k
        ax.scatter(coords[mask,0], coords[mask,1], s=11, color=color,
                   alpha=.52, edgecolors="none", rasterized=True, zorder=1)
    ax.scatter(sample[:,0], sample[:,1], s=105, marker="o", color="#E53935",
               edgecolors="black", linewidths=.9, zorder=5)
    ax.scatter(medoids[:,0], medoids[:,1], s=145, marker="x", color="black",
               linewidths=2.6, zorder=6)
    for k in range(n):
        ax.plot([medoids[k,0], assigned[k,0]], [medoids[k,1], assigned[k,1]],
                ls="--", lw=1.1, color=".20", alpha=.75, zorder=3)
    ax.set_title(f"{heading}\n{description};  DI = {di:.3f}", fontsize=11.5)
    ax.set_xlim(-.08, 1.14); ax.set_ylim(-.12, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$c_1$"); ax.set_ylabel(r"$c_2$", rotation=0, labelpad=12)
    ax.grid(alpha=.22)

legend = [
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#E53935",
           markeredgecolor="black", markersize=8, label="Observed sample"),
    Line2D([0],[0], marker="x", color="black", linestyle="none", markeredgewidth=2.3,
           markersize=9, label="Medoid"),
]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5,.955), ncol=3,
           frameon=False, fontsize=10)
fig.suptitle("DI under Sample Translation and Rotation in a 2D Population",
             fontsize=15, y=.995)
fig.subplots_adjust(top=.86, bottom=.06, left=.055, right=.985, wspace=.12, hspace=.22)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=320, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
for heading, angle, shift, di in outcomes:
    print(f"{heading}: angle={angle}, shift=({shift[0]:.6f},{shift[1]:.6f}), DI={di:.9f}")
# Compare partitions after optimally relabelling cluster IDs.
def changed_after_relabelling(labels_a, labels_b):
    overlap = np.array([[np.sum((labels_a == i) & (labels_b == j)) for j in range(n)] for i in range(n)])
    i, j = linear_sum_assignment(-overlap)
    return N - overlap[i, j].sum()

for i in range(len(all_labels)):
    for j in range(i + 1, len(all_labels)):
        print(f"Partition change {specs[i][0]} vs {specs[j][0]}: "
              f"{changed_after_relabelling(all_labels[i], all_labels[j])}/{N} units")
print("Saved:", OUT)
print("Saved:", OUT_PDF)
