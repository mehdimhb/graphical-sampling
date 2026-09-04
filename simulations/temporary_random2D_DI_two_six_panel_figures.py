"""Two paper-style 2x3 DI figures on one uniform-random 2D population."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.index import DensityDisparity, Moran, Voronoi, LocalBalance

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "simulations/results/temporary_2D_relative_rotation"
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
N = len(coords)


class SampleInitializedOT(FIPBalancedNMeans):
    def _get_labels_centroids(self, x, p, init_centroids=None):
        supplied = np.asarray(init_centroids, float)
        assert np.array_equal(supplied, self._supplied)
        return self._get_labels_centroids_ot_kmeans(x, p, supplied.copy())

    def fit_sample(self, population, sample):
        self._supplied = np.asarray(sample, float).copy()
        self.fit(population, init_centroids=self._supplied)
        return self


def make_population(n):
    return SimpleNamespace(
        coords=np.ascontiguousarray(coords), inclusions=np.full(N, n/N),
        indices=np.arange(N), ids=np.arange(1, N+1), variable=np.zeros(N), N=N, n=n,
    )


def fit_ot(population, sample):
    # The package OT helper uses Sinkhorn epsilon=0.01.
    return SampleInitializedOT(n=population.n, init_clust_method="ot").fit_sample(population, sample)


def reference_configuration(population):
    n = population.n
    if n == 4:
        # One initial/reference location in each spatial quadrant.
        centers = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75]])
    elif n == 6:
        # Six spatially distributed locations in a 3-by-2 arrangement.
        centers = np.array([
            [.18, .25], [.50, .25], [.82, .25],
            [.18, .75], [.50, .75], [.82, .75],
        ])
    elif n == 20:
        # Twenty spatially distributed locations in a 5-by-4 arrangement.
        cx, cy = np.meshgrid(np.linspace(.10, .90, 5), np.linspace(.14, .86, 4))
        centers = np.column_stack([cx.ravel(), cy.ravel()])
    else:
        centers = np.column_stack([np.full(n, .5), np.linspace(.10, .90, n)])
    scorer = DensityDisparity(population, representative="nmedoids", n_jobs=1)
    for _ in range(12):
        fit = fit_ot(population, centers)
        medoids, _ = scorer._cluster_medoids(fit.labels, fit.centroids)
        if np.allclose(centers, medoids):
            break
        centers = medoids
    return medoids


def nearest_distinct_population_units(targets):
    rows, cols = linear_sum_assignment(cdist(targets, coords))
    sample = np.empty_like(targets)
    sample[rows] = coords[cols]
    sample_indices = np.empty(len(targets), dtype=int)
    sample_indices[rows] = cols
    return sample, sample_indices


def transformed_sample(refs, angle, shift):
    center = refs.mean(axis=0)
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    targets = center + (refs-center) @ rotation.T + shift
    return nearest_distinct_population_units(targets)


def evaluate(population, sample, sample_indices):
    fit = fit_ot(population, sample)
    scorer = DensityDisparity(population, representative="nmedoids", n_jobs=1)
    medoids, _ = scorer._cluster_medoids(fit.labels, fit.centroids)
    rows, cols = linear_sum_assignment(cdist(medoids, sample))
    assigned = np.empty_like(medoids); assigned[rows] = sample[cols]
    translated = coords + (assigned-medoids)[fit.labels]
    di = scorer._score_single_density(scorer._density(translated))
    mi = float(Moran(population, method="tille").score(sample_indices[None, :])[0])
    vi = float(Voronoi(population).score(sample_indices)[0])
    bi = float(LocalBalance(population).score(sample_indices)[0])
    return fit.labels, medoids, assigned, float(di), mi, vi, bi


# Exactly one 20-by-20 grid step right and one step down.
shift_1 = np.array([1/19, -1/19])


def make_figure(n, rotation_angle):
    population = make_population(n)
    refs = reference_configuration(population)
    if n == 4:
        # Label order is lower-left, lower-right, upper-left, upper-right.
        palette = ["#F2BE2C", "#2878B5", "#E15759", "#59A14F"]
    elif n == 6:
        palette = ["#F2BE2C", "#2878B5", "#8E63CE",
                   "#E15759", "#59A14F", "#F28E5B"]
    else:
        palette = plt.colormaps["tab20"](np.linspace(0, 1, n))
    specs = [
        ("(a) Original sample", 0, np.zeros(2), "Starting configuration"),
        ("(b) Uniformly shifted sample", 0, shift_1, "Same shift applied to every sample point"),
        (r"(c) Sample rotated $90^\circ$", 90, np.zeros(2),
         "Counterclockwise about the sample center"),
        (r"(d) Sample rotated $45^\circ$", 45, np.zeros(2),
         "Counterclockwise about the sample center"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 19), sharex=True, sharey=True)
    results, labels_all = [], []
    original_sample, _ = transformed_sample(refs, 0, np.zeros(2))
    for panel_idx, (ax, (heading, angle, shift, description)) in enumerate(zip(axes.flat, specs)):
        sample, sample_indices = transformed_sample(refs, angle, shift)
        labels, medoids, assigned, di, mi, vi, bi = evaluate(population, sample, sample_indices)
        results.append((heading, di, mi, vi, bi)); labels_all.append(labels)
        for k, color in enumerate(palette):
            mask = labels == k
            cluster_points = coords[mask]
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                boundary = cluster_points[np.r_[hull.vertices, hull.vertices[0]]]
                ax.fill(boundary[:,0], boundary[:,1], facecolor=color,
                        edgecolor="none", alpha=.10, zorder=0)
            ax.scatter(cluster_points[:,0], cluster_points[:,1], s=16, color=color,
                       alpha=.92, edgecolors="none", rasterized=True, zorder=1)
        ax.scatter(original_sample[:,0], original_sample[:,1], s=138, marker="o",
                   facecolors="none", edgecolors=".38", linewidths=1.7, zorder=4)
        if panel_idx > 0:
            # One representative arrow keeps the n=20 panels readable.
            representative = 12 if panel_idx == 1 else 6
            old, new = original_sample[representative], sample[representative]
            curved = angle != 0
            ax.annotate("", xy=new, xytext=old,
                        arrowprops=dict(arrowstyle="-|>", color="#008C95", lw=2.2,
                                        linestyle="-", mutation_scale=17,
                                        shrinkA=6, shrinkB=8,
                                        connectionstyle="arc3,rad=.28" if curved else "arc3,rad=0"),
                        zorder=7)
            if curved:
                ax.text(.50, .52, rf"${angle}^\circ$", color="#00747B",
                        fontsize=20, fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=.18", facecolor="white",
                                  edgecolor="none", alpha=.82), zorder=8)
        ax.scatter(sample[:,0], sample[:,1], s=92, marker="o", color="#E53935",
                   edgecolors="black", linewidths=.9, zorder=5)
        ax.scatter(medoids[:,0], medoids[:,1], s=125, marker="x", color="black",
                   linewidths=1.9, zorder=6)
        for k in range(n):
            ax.plot([medoids[k,0], assigned[k,0]], [medoids[k,1], assigned[k,1]],
                    ls=(0, (5, 2.5)), lw=1.7, color=".08", alpha=.92, zorder=3)
        displayed_di = 0.0 if abs(di) < 0.0005 else di
        ax.set_title(
            f"{heading}\n{description}\n"
            f"DI = {displayed_di:.2f};  MI = {mi:.2f};  VI = {vi:.2f};  BI = {bi:.2f}",
            fontsize=21, fontweight="bold", linespacing=1.14,
        )
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$c_1$")
        ax.set_ylabel(r"$c_2$", rotation=0, labelpad=12)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.tick_params(axis="both", labelsize=17)
        ax.grid(False)
    legend = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#E53935",
               markeredgecolor="black", markersize=8, label="Observed sample after transformation"),
        Line2D([0],[0], marker="o", color=".38", markerfacecolor="none",
               markeredgewidth=1.7, linestyle="none", markersize=10,
               label="Original sample position"),
        Line2D([0],[0], color="#008C95", linestyle="-", lw=1.8,
               marker=">", markerfacecolor="#008C95", markeredgecolor="#008C95",
               markersize=6, label="Original-to-transformed movement"),
        Line2D([0],[0], marker="x", color="black", linestyle="none",
               markeredgewidth=1.8, markersize=9, label="IPnM medoid (reference unit)"),
        Line2D([0],[0], color=".08", linestyle=(0, (5, 2.5)), lw=1.7,
               label="Optimal medoid–sample match"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5,.915),
               ncol=3, frameon=False,
               prop={"size": 16, "weight": "bold"},
               handletextpad=.7, columnspacing=1.7)
    fig.suptitle(rf"Translation and Rotation Properties of DI ($n={n}$)",
                 fontsize=32, fontweight="bold", y=.995)
    fig.subplots_adjust(top=.75, bottom=.055, left=.07, right=.98,
                        wspace=.10, hspace=.42)
    png = OUTDIR / f"grid_20x20_N400_DI_original_shift_rotation90_rotation45_large_2x2_n{n}.png"
    pdf = png.with_suffix(".pdf")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"n={n}")
    for heading, di, mi, vi, bi in results:
        print(f"  {heading}: DI={di:.9f}, MI={mi:.9f}, VI={vi:.9f}, BI={bi:.9f}")
    # Verify that each sample really produced its own final partition.
    def changed(a, b):
        overlap = np.array([[np.sum((a==i)&(b==j)) for j in range(n)] for i in range(n)])
        i, j = linear_sum_assignment(-overlap)
        return N-overlap[i,j].sum()
    print("  changed units: original vs shift =", changed(labels_all[0], labels_all[1]))
    print("  changed units: original vs rotation90 =", changed(labels_all[0], labels_all[2]))
    print("  changed units: original vs rotation45 =", changed(labels_all[0], labels_all[3]))
    print("Saved:", png)
    print("Saved:", pdf)


make_figure(20, 45)
