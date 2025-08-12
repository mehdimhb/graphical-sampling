from itertools import pairwise
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError

from .border import Borders
from ..clustering import UPBalancedKMeans

@dataclass
class Zone:
    units: NDArray

@dataclass
class Cluster:
    units: NDArray
    zones: list[Zone]

class Organizer:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        n_clusters: int,
        n_zones: tuple[int, int],
        tolerance: int,
        split_size: float,
        zone_mode: str = "sweep",
        sort_method: str = "lexico",
        zonal_sort: str = "lexico"
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.n_clusters = n_clusters
        self.n_zones = n_zones
        self.tolerance = tolerance
        self.split_size = split_size
        self.zone_mode = zone_mode
        self.sort_method = sort_method
        self.zonal_sort = zonal_sort
        self.dbk = None
        self.clusters = self._generate_clusters()
        self.borders = self._generate_borders()

    def _generate_borders(self):
        border_ids = set(np.where((self.dbk.membership > 0).sum(axis=1) > 1)[0])
        borders = Borders(border_ids)
        borders.build_from(self.clusters)
        return borders

    def _generate_clusters(self) -> list[Cluster]:
        self.dbk = AuxiliaryBalancedKMeans(k=self.n_clusters, split_size=self.split_size)
        self.dbk.fit(self.coords, self.probs)
        return [
            Cluster(units=units, zones=self._generate_zones(units))
            for units in self.dbk.clusters
        ]

    def _generate_zones(self, units: NDArray) -> list[Zone]:
        match self.zone_mode:
            case "cluster":
                return self._generate_zones_with_cluster(units)
            case "sweep":
                return self._generate_zones_with_sweep(units)
        return [Zone(units=units)]

    def _generate_zones_with_cluster(self, units: NDArray) -> list[Zone]:
        if np.prod(self.n_zones) > 1:
            dbk = AuxiliaryBalancedKMeans(k=np.prod(self.n_zones), split_size=self.split_size)
            dbk.fit(coords=units[:, 1:3], probs=units[:, 3], population_ids=units[:, 0])
            clusters = dbk.clusters
        else:
            clusters = [units]
        zones = []
        for zone_units in clusters:
            zone_units[:, 3] = self._numerical_stabilizer(zone_units[:, 3])
            idx = self._sort_func(zone_units[:, 1:3], self.sort_method)
            zone_units = zone_units[idx]
            zones.append(Zone(units=zone_units))
        return self._sort_zones(zones)

    def _generate_zones_with_sweep(self, units) -> list[Zone]:
        vertical_zones = self._sweep(
            units[np.argsort(units[:, 1])], 1 / self.n_zones[0]
        )
        zones = []
        for zone in vertical_zones:
            units_of_basic_zones = self._sweep(
                zone[np.argsort(zone[:, 2])], 1 / (np.prod(self.n_zones))
            )
            for zone_units in units_of_basic_zones:
                zone_units[:, 3] = self._numerical_stabilizer(zone_units[:, 3])
                idx = self._sort_func(zone_units[:, 1:3], self.sort_method)
                zone_units = zone_units[idx]
                zones.append(Zone(units=zone_units))
        return self._sort_zones(zones)

    def _sort_zones(self, zones: list[Zone]) -> list[Zone]:
        zones_centroids = np.array([
            zone.units[:, 1:3].mean(axis=0) for zone in zones
        ])
        sorted_zones = np.array(zones)
        idx = self._sort_func(self._normalize(np.array(zones_centroids)), self.zonal_sort)
        sorted_zones = sorted_zones[idx]
        sorted_zones = sorted_zones.tolist()
        return sorted_zones

    def _normalize(self, coords: np.ndarray) -> np.ndarray:
        return (coords - coords.min(axis=0)) / (np.ptp(coords, axis=0) + 1e-6)

    def _sort_func(self, units: NDArray, method: str) -> NDArray:
        match method:
            case "lexico-yx":
                return np.lexsort((units[:, 1], units[:, 0]))
            case "lexico-xy":
                return np.lexsort((units[:, 0], units[:, 1]))
            case "random":
                return np.random.permutation(units.shape[0])
            case "angle_0":
                angles = np.mod(np.arctan2(units[:, 1], units[:, 0]), 2 * np.pi)
                return np.argsort(angles)
            case "distance_0":
                return np.argsort(np.linalg.norm(units, axis=1))
            case "projection":
                return np.argsort(units[:, 0] + units[:, 1])
            case "center":
                centroid = units.mean(axis=0)
                return np.argsort(np.linalg.norm(units - centroid, axis=1))
            case "spiral":
                return spiral_sort_indices(units)
            case "max":
                return np.argsort(np.max(units, axis=1))

        return np.arange(units.shape[0])

    def _sweep(self, units: NDArray, threshold: float) -> list[NDArray]:
        boarder_units_remainings, zones_indices = self._generate_boarders_and_indices(
            units, threshold
        )
        swept_zones = []
        for indices in pairwise(zones_indices):
            zone, boarder_units_remainings = self._sweep_zone(
                units, boarder_units_remainings, indices, threshold
            )
            swept_zones.append(zone)
        return swept_zones

    def _generate_boarders_and_indices(self, units: NDArray, threshold: float):
        thresholds = np.arange(
            threshold, np.sum(units[:, 3]) - threshold / 2, threshold
        )
        indices = np.concatenate(
            (
                [0],
                np.searchsorted(units[:, 3].cumsum(), thresholds, side="right"),
                [units.shape[0] - 1],
            )
        )
        boarder_units = {index: units[index][3] for index in np.unique(indices)}
        return boarder_units, indices

    def _sweep_zone(
        self,
        units: NDArray,
        boarder_units_remainings: NDArray,
        indices: tuple[NDArray, NDArray],
        threshold: float,
    ) -> NDArray:
        zone, start_remainder = self._sweep_boarder_unit(
            np.array([]).reshape(0, 4),
            units[indices[0]],
            boarder_units_remainings[indices[0]],
            threshold,
        )
        boarder_units_remainings[indices[0]] = start_remainder

        zone = np.concatenate([zone, units[indices[0] + 1 : indices[1]]])

        zone, stop_remainder = self._sweep_boarder_unit(
            zone,
            units[indices[1]],
            boarder_units_remainings[indices[1]],
            threshold - np.sum(zone[:, 3]),
        )
        boarder_units_remainings[indices[1]] = stop_remainder

        return zone, boarder_units_remainings

    def _sweep_boarder_unit(
        self, zone: NDArray, unit: NDArray, probability: float, threshold: float
    ) -> tuple[NDArray, float]:
        if probability < 10**-self.tolerance:
            return zone, 0
        if threshold < 10**-self.tolerance:
            return zone, probability
        if probability < threshold - 10**-self.tolerance:
            return np.concatenate(
                [zone, np.append(unit[:3], probability).reshape(1, -1)]
            ), 0
        elif probability > threshold + 10**-self.tolerance:
            return np.concatenate(
                [zone, np.append(unit[:3], threshold).reshape(1, -1)]
            ), probability - threshold
        return np.concatenate([zone, np.append(unit[:3], threshold).reshape(1, -1)]), 0

    def _numerical_stabilizer(self, probs: NDArray) -> NDArray:
        probs_stabled = np.round(probs, self.tolerance)
        probs_stabled *= 1 / (np.sum(probs_stabled) * np.prod(self.n_zones))
        return probs_stabled

    @staticmethod
    def get_sorted_cluster_indices_by_lexico(centroids, jitter=1e-8):
        noise = np.random.uniform(-jitter, jitter, size=centroids.shape)
        perturbed = centroids + noise
        sorted_order = np.lexsort((perturbed[:, 0], perturbed[:, 1]))
        label_to_color = np.zeros_like(sorted_order)
        for color_idx, label in enumerate(sorted_order):
            label_to_color[label] = color_idx
        return label_to_color, centroids[sorted_order], sorted_order

    def plot(
        self,
        ax=None,
        figsize: tuple[int, int] = (8, 6),
        background_gdf=None,
    ) -> None:
        # Color palette (sort order)
        bardi_colors = [
            "#4CAF50",  # 1. Green
            "#2196F3",  # 2. Blue
            "#F44336",  # 3. Red
            "#FFEB3B",  # 4. Yellow
            "#FF9800",  # 5. Orange
            "#9C27B0",  # 6. Violet
            "#E91E63",  # 7. Pink/Magenta
            "#00BCD4",  # 8. Turquoise/Cyan
            "#BDBDBD",  # 9. Medium Gray
            "#FFD700"   # 10. Gold/Amber (strong yellow-gold)
        ]

        def plot_convex_hull(points, ax, color, alpha=0.33, edge_color="gray", line_width=0.6):
            points = np.asarray(points)
            if (points.ndim != 2) or (points.shape[0] < 3) or (points.shape[1] != 2):
                return ax, None
            if np.allclose(points[:,0], points[0,0]) or np.allclose(points[:,1], points[0,1]):
                return ax, None
            try:
                hull = ConvexHull(points)
                polygon = Polygon(
                    points[hull.vertices],
                    closed=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor=edge_color,
                    lw=line_width,
                    zorder=1
                )
                ax.add_patch(polygon)
                return ax, hull
            except QhullError:
                return ax, None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Draw background
        if background_gdf is not None:
            background_gdf.plot(
                ax=ax, color="white", edgecolor="black", linewidth=1.5, zorder=0
            )

        # ---- SORTING AND COLORING CLUSTERS!!! ----
        n_clusters = len(self.clusters)
        # Compute centroids for all clusters
        centroids = np.zeros((n_clusters, 2))
        for i, cluster in enumerate(self.clusters):
            centroids[i] = cluster.units[:, 1:3].mean(axis=0) if len(cluster.units) > 0 else np.nan

        # Get sorted indices and color mapping
        #label_to_color, sorted_centroids = self.get_sorted_cluster_indices_by_lexico(centroids)

        # ------------- PLOTTING LOOP ----------------
        label_to_color, sorted_centroids, sorted_order = self.get_sorted_cluster_indices_by_lexico(centroids)
        for plot_idx, cluster_idx in enumerate(sorted_order):
            cluster = self.clusters[cluster_idx]
            cluster_color = bardi_colors[plot_idx % len(bardi_colors)]
            cluster_points = cluster.units[:, 1:3]
    # (rest of your plotting code as above)
            # Cluster convex hull
            ax, _ = plot_convex_hull(
                cluster_points,
                ax,
                color=cluster_color,
                alpha=0.36,
                edge_color="black",
                line_width=0.9
            )
            # Units scatter
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=cluster_color,
                edgecolors="none",
                s=cluster.units[:, 3] * 1000,
                alpha=0.88,
                zorder=2,
            )
            # Label
            prob_sum = round(cluster.units[:, 3].sum(), 2)
            center = cluster_points.mean(axis=0)
            # ax.text(
            #     center[0],
            #     center[1],
            #     f"{prob_sum}",
            #     color="black",
            #     fontsize=10,
            #     weight="bold",
            #     alpha=0.85,
            #     ha="center",
            #     va="center",
            #     zorder=3,
            # )

            # Zones (match cluster color)
            for zone_idx, zone in enumerate(cluster.zones):
                zone_points = zone.units[:, 1:3]
                zone_color = cluster_color
                ax, hull = plot_convex_hull(
                    zone_points,
                    ax,
                    color=zone_color,
                    alpha=0.17,
                    edge_color="gray",
                    line_width=0.5,
                )
                hull_center = np.mean(
                    zone_points if hull is None else zone_points[hull.vertices], axis=0
                )
                ax.text(
                    hull_center[0],
                    hull_center[1],
                    f"{zone_idx + 1}",
                    color="black",
                    fontsize=13,
                    alpha=0.22,
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=4,
                )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")
        return ax

    def plot_with_samples(self, samples: NDArray, max_cols: int = 4) -> None:
        n_samples = len(samples)
        n_cols = min(max_cols, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        figsize = (5 * n_cols, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_samples > 1 else [axes]

        for sample_idx, sample in enumerate(samples):
            ax = axes[sample_idx]
            ax = self.plot(ax)
            ax.scatter(
                self.coords[sample][:, 0],
                self.coords[sample][:, 1],
                color="black",
                marker="X",
                alpha=0.8,
                s=100,
                label=f"Sample {sample_idx + 1}",
                zorder=5
            )
            ax.set_title(f"Sample {sample_idx + 1}")

        for ax in axes[n_samples:]:
            fig.delaxes(ax)


def convex_hull_indices(points):
    """
    Andrew's monotonic chain algorithm.
    Input: points as an array of shape (N, 2)
    Output: list of indices of points forming the convex hull in CCW order.
    """
    N = len(points)
    if N < 3:
        return list(range(N))

    # Sort points lexicographically by x, then y
    sorted_idx = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[sorted_idx]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for idx in sorted_idx:
        while len(lower) >= 2 and cross(points[lower[-2]], points[lower[-1]], points[idx]) <= 0:
            lower.pop()
        lower.append(idx)

    upper = []
    for idx in reversed(sorted_idx):
        while len(upper) >= 2 and cross(points[upper[-2]], points[upper[-1]], points[idx]) <= 0:
            upper.pop()
        upper.append(idx)

    # Concatenate lower and upper, excluding last point of each (it's repeated)
    hull = lower[:-1] + upper[:-1]
    return hull


def spiral_sort_indices(points):
    """
    Peel convex hull layers off the set of points until none remain.
    Returns the concatenated list of all indices in peel order.
    """
    remaining = list(range(len(points)))
    order = []

    while remaining:
        subpts = points[remaining]
        hull = convex_hull_indices(subpts)
        # Map hull indices within subpts back to original indices
        hull_orig = [remaining[i] for i in hull]

        # Find start: bottom-left on hull
        coords_hull = points[hull_orig]
        start = np.lexsort((coords_hull[:, 0], coords_hull[:, 1]))[0]
        # Rotate hull list to start here
        rotated = hull_orig[start:] + hull_orig[:start]

        # Append and remove these indices
        order.extend(rotated)
        for idx in rotated:
            remaining.remove(idx)

    return order