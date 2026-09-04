"""Targeted 2D check: rotate a shifted sample relative to a fixed population."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.index import DensityDisparity

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "simulations/results/temporary_2D_relative_rotation"
OUTPNG = OUTDIR / "2D_DI_shift_plus_relative_rotation_n3_n4_n5.png"
OUTCSV = OUTDIR / "2D_DI_shift_plus_relative_rotation_n3_n4_n5.csv"

# Same regular-square population geometry as the sample-adaptive illustration,
# made denser so n=3, 4, and 5 can all have exact equal-size hard clusters.
xx, yy = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 20))
coords = np.column_stack([xx.ravel(), yy.ravel()])
N = len(coords)
shift = np.array([0.08, -0.05])
angles = np.arange(-90, 91, 15)


class SampleInitializedOT(FIPBalancedNMeans):
    """Force the supplied sample coordinates into the epsilon=.01 OT solver."""
    def _get_labels_centroids(self, x, p, init_centroids=None):
        supplied = np.asarray(init_centroids, dtype=float)
        assert np.array_equal(supplied, self._supplied_centers)
        return self._get_labels_centroids_ot_kmeans(x, p, supplied.copy())

    def fit_from_sample(self, population, sample):
        self._supplied_centers = np.asarray(sample, dtype=float).copy()
        return self.fit(population, init_centroids=self._supplied_centers)


def population_for(n):
    return SimpleNamespace(
        coords=np.ascontiguousarray(coords), inclusions=np.full(N, n / N),
        indices=np.arange(N), ids=np.arange(N), variable=np.zeros(N), N=N, n=n,
    )


def initial_polygon(n):
    # Only establishes a stable first OT partition; it is not reused in DI runs.
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return np.column_stack([0.5 + 0.32 * np.cos(a), 0.5 + 0.32 * np.sin(a)])


def fit_ot(pop, sample):
    # The package OT helper constructs SinkhornKMeans with epsilon=0.01.
    fit = SampleInitializedOT(n=pop.n, init_clust_method="ot")
    fit.fit_from_sample(pop, sample)
    return fit


def reference_medoids(pop):
    scorer = DensityDisparity(pop, representative="nmedoids", n_jobs=1)
    centers = initial_polygon(pop.n)
    # Iterate sample-initialized fits to a stable medoid reference configuration.
    for _ in range(12):
        fit = fit_ot(pop, centers)
        medoids, _ = scorer._cluster_medoids(fit.labels, fit.centroids)
        if np.allclose(medoids, centers):
            break
        centers = medoids
    return medoids


def rotate(points, center, degrees):
    a = np.deg2rad(degrees)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return center + (points - center) @ R.T


def calculate_di(pop, sample):
    # Paper-defined path: each realized sample initializes its own OT partition.
    fit = fit_ot(pop, sample)
    scorer = DensityDisparity(pop, representative="nmedoids", n_jobs=1)
    medoids, _ = scorer._cluster_medoids(fit.labels, fit.centroids)
    rows, cols = linear_sum_assignment(cdist(medoids, sample))
    assigned = np.empty_like(medoids)
    assigned[rows] = sample[cols]
    translated = coords + (assigned - medoids)[fit.labels]
    di = scorer._score_single_density(scorer._density(translated))
    return float(di)


records = []
for n in (3, 4, 5):
    pop = population_for(n)
    refs = reference_medoids(pop)
    pivot = refs.mean(axis=0)
    for angle in angles:
        # Population stays fixed. The medoid pattern is rotated first, and the
        # same translation vector is then applied to every sample center.
        sample = rotate(refs, pivot, angle) + shift
        records.append({"n": n, "angle_degrees": angle, "DI": calculate_di(pop, sample)})

results = pd.DataFrame(records)
OUTDIR.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTCSV, index=False)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
for ax, n, color in zip(axes, (3, 4, 5), ("#168BD2", "#FF7A00", "#24A148")):
    d = results[results.n == n]
    ax.plot(d.angle_degrees, d.DI, marker="o", ms=5, lw=1.8, color=color)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.axvline(0, color="0.45", lw=1, ls=":")
    ax.set_title(rf"Sample size $n={n}$")
    ax.set_xlabel(r"Relative rotation angle $\theta$")
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.grid(alpha=.25)
axes[0].set_ylabel("DI")
fig.suptitle("DI after a Common Shift and Rotation of the Sample Relative to the Population", y=1.02)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")

print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("Saved plot:", OUTPNG)
print("Saved values:", OUTCSV)
