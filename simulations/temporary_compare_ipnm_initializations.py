"""Temporary, read-only-method comparison of OT and expanded IPnM DI evaluation.

This script does not modify package code or existing results. It writes a new,
isolated results folder under simulations/results/ipnm_ot_vs_expanded/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.index import DensityDisparity
from graphical_sampling.population import Population


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "simulations/results/ipnm_ot_vs_expanded"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

M = 200
SEED = 20260903
SIGN_TOL = 0.01
PARTITION_CHECK_SAMPLES = 5

EXPERIMENTS = {
    "grid": (ROOT / "simulations/populations/simulated/grid_N144_perm.csv", [4, 16]),
    "random": (ROOT / "simulations/populations/simulated/rand_N144_perm.csv", [4, 16]),
    "clustered": (ROOT / "simulations/populations/simulated/clust_N144_perm.csv", [4, 16]),
    "meuse": (ROOT / "simulations/populations/real/meuse.csv", [5, 20]),
}


def load_population(path: Path, population_name: str, n: int, inclusion: str):
    df = pd.read_csv(path)
    coords = df[["x", "y"]].to_numpy(float)
    if inclusion == "EP":
        weights = np.ones(len(df), dtype=float)
    elif population_name == "meuse":
        weights = df["copper"].to_numpy(float)
    else:
        weights = df["prob"].to_numpy(float)

    variable_col = "cadmium" if population_name == "meuse" else "z.90"
    pop = Population(
        coords=coords,
        inclusions=weights,
        variable=df[variable_col].to_numpy(float),
        n=n,
    )
    return pop


def maxentropy_samples(pik: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Draw fixed-size samples; one matrix is reused by both DI methods."""
    N = len(pik)
    n = int(round(pik.sum()))
    samples = np.empty((m, n), dtype=int)
    ro.r("suppressPackageStartupMessages(library(sampling))")
    ro.r["set.seed"](seed)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["pik_compare"] = np.asarray(pik, dtype=float)
        for i in range(m):
            out = np.asarray(ro.r("sampling::UPmaxentropy(pik_compare)")).ravel()
            idx = np.flatnonzero(out > 0.5) if out.size == N else out.astype(int) - 1
            if len(idx) != n or len(np.unique(idx)) != n:
                raise RuntimeError(f"Invalid sample at iteration {i}.")
            samples[i] = idx
    return samples


def fit_partition(pop: Population, method: str):
    """Fit exactly the requested FIPBalancedNMeans branch."""
    fit = FIPBalancedNMeans(
        n=pop.n,
        init_clust_method=method,
        split_size=0.001,
    )
    fit.fit(pop)
    return fit.labels, fit.centroids


def weighted_medoids(coords, pik, labels, n_clusters):
    medoids = np.empty((n_clusters, coords.shape[1]), dtype=float)
    indices = np.empty(n_clusters, dtype=int)
    for k in range(n_clusters):
        idx = np.flatnonzero(labels == k)
        distances = cdist(coords[idx], coords[idx])
        objective = distances @ pik[idx]
        best = int(np.argmin(objective))
        indices[k] = idx[best]
        medoids[k] = coords[indices[k]]
    return medoids, indices


def score_with_reference(sample, coords, labels, medoids, scorer):
    raw_sample = coords[sample]
    rows, cols = linear_sum_assignment(cdist(medoids, raw_sample))
    assigned = np.empty_like(medoids)
    assigned[rows] = raw_sample[cols]
    translated = coords + (assigned - medoids)[labels]
    return scorer._score_single_density(scorer._density(translated))


def sign_class(values):
    values = np.asarray(values)
    return np.where(values < -SIGN_TOL, "negative", np.where(values > SIGN_TOL, "positive", "near_zero"))


def coassignment_disagreement(labels_a, labels_b):
    same_a = labels_a[:, None] == labels_a[None, :]
    same_b = labels_b[:, None] == labels_b[None, :]
    upper = np.triu_indices(len(labels_a), k=1)
    return np.mean(same_a[upper] != same_b[upper])


raw_rows = []
partition_rows = []
setting_number = 0

for population_name, (path, sample_sizes) in EXPERIMENTS.items():
    for inclusion in ["EP", "UP"]:
        for n in sample_sizes:
            setting_number += 1
            print(f"[{setting_number}/16] {population_name}, {inclusion}, n={n}")
            pop = load_population(path, population_name, n, inclusion)
            samples = maxentropy_samples(pop.inclusions, M, SEED + 1000 * setting_number)
            scorer = DensityDisparity(population=pop, representative="nmedoids", n_jobs=1)

            references = {}
            for method in ["ot", "expanded"]:
                labels, _ = fit_partition(pop, method)
                medoids, medoid_indices = weighted_medoids(pop.coords, pop.inclusions, labels, pop.n)
                references[method] = {
                    "labels": labels,
                    "medoids": medoids,
                    "medoid_indices": medoid_indices,
                }

            ot_labels = references["ot"]["labels"]
            ex_labels = references["expanded"]["labels"]
            ot_medoids = references["ot"]["medoids"]
            ex_medoids = references["expanded"]["medoids"]
            medoid_rows, medoid_cols = linear_sum_assignment(cdist(ot_medoids, ex_medoids))
            medoid_set_ot = set(references["ot"]["medoid_indices"].tolist())
            medoid_set_ex = set(references["expanded"]["medoid_indices"].tolist())
            medoid_union = medoid_set_ot | medoid_set_ex

            common_partition_metrics = {
                "adjusted_rand_index": adjusted_rand_score(ot_labels, ex_labels),
                "coassignment_disagreement": coassignment_disagreement(ot_labels, ex_labels),
                "medoid_set_jaccard": len(medoid_set_ot & medoid_set_ex) / len(medoid_union),
                "mean_matched_medoid_distance": cdist(ot_medoids, ex_medoids)[medoid_rows, medoid_cols].mean(),
                "identical_partition_up_to_labels": bool(adjusted_rand_score(ot_labels, ex_labels) == 1.0),
                "identical_medoid_set": medoid_set_ot == medoid_set_ex,
            }

            for iteration, sample in enumerate(tqdm(samples, leave=False), start=1):
                di_ot = score_with_reference(sample, pop.coords, ot_labels, ot_medoids, scorer)
                di_expanded = score_with_reference(sample, pop.coords, ex_labels, ex_medoids, scorer)
                raw_rows.append(
                    {
                        "population": population_name,
                        "inclusion": inclusion,
                        "n": n,
                        "iteration": iteration,
                        "sample_indices": " ".join(map(str, sample.tolist())),
                        "DI_ot": di_ot,
                        "DI_expanded": di_expanded,
                        "absolute_difference": abs(di_ot - di_expanded),
                    }
                )
                if iteration <= PARTITION_CHECK_SAMPLES:
                    partition_rows.append(
                        {
                            "population": population_name,
                            "inclusion": inclusion,
                            "n": n,
                            "iteration": iteration,
                            "sample_indices": " ".join(map(str, sample.tolist())),
                            **common_partition_metrics,
                        }
                    )

raw = pd.DataFrame(raw_rows)
raw["sign_ot"] = sign_class(raw["DI_ot"])
raw["sign_expanded"] = sign_class(raw["DI_expanded"])
raw["same_sign"] = raw["sign_ot"] == raw["sign_expanded"]

summary_rows = []
group_cols = ["population", "inclusion", "n"]
for keys, group in raw.groupby(group_cols, sort=False):
    correlation = group[["DI_ot", "DI_expanded"]].corr().iloc[0, 1]
    summary_rows.append(
        {
            **dict(zip(group_cols, keys)),
            "M": len(group),
            "mean_DI_ot": group["DI_ot"].mean(),
            "mean_DI_expanded": group["DI_expanded"].mean(),
            "sd_DI_ot": group["DI_ot"].std(ddof=1),
            "sd_DI_expanded": group["DI_expanded"].std(ddof=1),
            "mean_absolute_difference": group["absolute_difference"].mean(),
            "median_absolute_difference": group["absolute_difference"].median(),
            "maximum_absolute_difference": group["absolute_difference"].max(),
            "correlation": correlation,
            "same_sign_proportion": group["same_sign"].mean(),
        }
    )

summary = pd.DataFrame(summary_rows)
partitions = pd.DataFrame(partition_rows)

raw.to_csv(OUTPUT_DIR / "sample_level_DI_ot_vs_expanded.csv", index=False)
summary.to_csv(OUTPUT_DIR / "summary_DI_ot_vs_expanded.csv", index=False)
partitions.to_csv(OUTPUT_DIR / "partition_medoid_checks.csv", index=False)

sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
for ax, population_name in zip(axes.flat, EXPERIMENTS):
    plot_data = raw[raw["population"] == population_name]
    sns.scatterplot(
        data=plot_data,
        x="DI_ot",
        y="DI_expanded",
        hue="inclusion",
        style="n",
        alpha=0.48,
        s=30,
        ax=ax,
    )
    lo = min(plot_data["DI_ot"].min(), plot_data["DI_expanded"].min())
    hi = max(plot_data["DI_ot"].max(), plot_data["DI_expanded"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2)
    ax.set_title(population_name.capitalize())
    ax.set_xlabel("DI (OT)")
    ax.set_ylabel("DI (expanded)")
    ax.set_aspect("equal", adjustable="box")

fig.suptitle("IPnM initialization comparison using identical realized samples", y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "scatter_DI_ot_vs_expanded.png", dpi=300, bbox_inches="tight")
plt.close(fig)

overall_mad = raw["absolute_difference"].mean()
overall_corr = raw[["DI_ot", "DI_expanded"]].corr().iloc[0, 1]
overall_sign = raw["same_sign"].mean()
if overall_mad < 0.01 and overall_corr >= 0.99:
    category = "numerically almost identical"
elif overall_mad < 0.05 and overall_corr >= 0.90:
    category = "strongly correlated but not identical"
else:
    category = "materially different"

conclusion = pd.DataFrame(
    [{
        "M_per_setting": M,
        "total_samples": len(raw),
        "overall_mean_absolute_difference": overall_mad,
        "overall_correlation": overall_corr,
        "overall_same_sign_proportion": overall_sign,
        "classification": category,
    }]
)
conclusion.to_csv(OUTPUT_DIR / "overall_conclusion.csv", index=False)

print("\nPer-setting summary")
print(summary.round(5).to_string(index=False))
print("\nOverall conclusion")
print(conclusion.round(5).to_string(index=False))
print(f"\nResults written to {OUTPUT_DIR}")
