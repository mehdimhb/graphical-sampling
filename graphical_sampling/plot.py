import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError
from matplotlib.patches import Polygon
from typing import List, Optional, Tuple

from .population import Population
from .sampling import Cluster

def plot(
    population: Population,
    clusters: List[Cluster],
    connect_points: bool = False,
    ax: Optional[plt.Axes] = None,
    background_gdf=None,
    size_scale: float = 1000.0,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 100
) -> plt.Axes:
    """Plot a population and its clusters with convex hulls and zone numbering."""

    def _draw_hull(points: np.ndarray, color: str, alpha: float, edge_color: str, lw: float):
        if points.shape[0] < 3:
            return None
        try:
            hull = ConvexHull(points)
            verts = points[hull.vertices]
        except (QhullError, ValueError):
            return None
        ax.add_patch(Polygon(verts, closed=True, facecolor=color, alpha=alpha,
                            edgecolor=edge_color, lw=lw))
        return verts.mean(axis=0)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if background_gdf is not None:
        background_gdf.plot(ax=ax, color="white", edgecolor="black", linewidth=1.5, zorder=0)

    # ---------- 1) weighted centroids per cluster ----------
    k = len(clusters)
    centroids = np.full((k, 2), np.nan, float)
    for i, cluster in enumerate(clusters):
        idx_share = cluster.get_index_share(reduce=True)
        if idx_share.size == 0:
            continue
        unit_idx = idx_share[:, 0].astype(int)
        shares   = idx_share[:, 1]
        pts      = population.coords[unit_idx]
        w        = population.probs[unit_idx] * shares
        s = float(w.sum())
        centroids[i] = (pts * w[:, None]).sum(axis=0) / s if s > 0 else pts.mean(axis=0)

    valid = ~np.isnan(centroids).any(axis=1)

    # ---------- 2) deterministic color assignment ----------
    # Fixed TL, TR, BL, BR (green, red, blue, amber)
    palette4    = ["#59A14F", "#E15759", "#1F77B4", "#FDD835"]
    palette_seq = ["#59A14F", "#E15759", "#1F77B4", "#FDD835",
                   "#9467BD", "#8C564B", "#17BECF", "#FF7F0E","#56B4E9",  # sky blue
    "#EF5350",  # teal
    "#CC79A7",  # magenta
    "#FBC02D",  # vermillion
    "#7F3C8D",  # deep purple
    "#11A579",  # green-teal
    "#EF5350",  # indigo
    "#FFC107"   # warm orange
    ]

    colors_for_cluster = ["#808080"] * k  # default gray

    if k == 4 and valid.all():
        # Split by vertical median into "top" & "bottom", then sort each by x
        y_mid = np.median(centroids[:, 1])

        top_idx = np.where(centroids[:, 1] >= y_mid)[0]
        bot_idx = np.where(centroids[:, 1] <  y_mid)[0]

        # If degeneracy (not 2+2), fall back to y-desc split
        if top_idx.size != 2 or bot_idx.size != 2:
            order_y = np.argsort(-centroids[:, 1])  # y desc
            top_idx = order_y[:2]; bot_idx = order_y[2:]

        tl = top_idx[np.argmin(centroids[top_idx, 0])]
        tr = top_idx[np.argmax(centroids[top_idx, 0])]
        bl = bot_idx[np.argmin(centroids[bot_idx, 0])]
        br = bot_idx[np.argmax(centroids[bot_idx, 0])]

        # Assign colors in strict TL, TR, BL, BR order — guarantees 4 distinct colors
        for idx, col in zip([tl, tr, bl, br], palette4):
            colors_for_cluster[idx] = col
    else:
        # Any other k: row-major (y desc, then x asc)
        if valid.any():
            order = np.lexsort((centroids[:, 0], -centroids[:, 1]))
        else:
            order = np.arange(k)
        for rank, idx in enumerate(order):
            colors_for_cluster[idx] = palette_seq[rank % len(palette_seq)]

    # ---------- 3) draw clusters ----------
    for i, cluster in enumerate(clusters):
        color = colors_for_cluster[i]
        idx_share = cluster.get_index_share(reduce=True)
        if idx_share.size == 0:
            continue

        unit_idx = idx_share[:, 0].astype(int)
        shares   = idx_share[:, 1]
        coords   = population.coords[unit_idx]
        probs    = population.probs[unit_idx] * shares

        _draw_hull(coords, color=color, alpha=0.20, edge_color="black", lw=1.0)

        sizes = probs * size_scale
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color=color,
                   edgecolors="none", alpha=1.0, zorder=2)

        for z, zone in enumerate(cluster.zones):
            z_idx = zone.index.astype(int)
            z_pts = population.coords[z_idx]
            if connect_points and z_pts.shape[0] > 1:
                ax.plot(z_pts[:, 0], z_pts[:, 1], linestyle="-", color=color, alpha=0.7, zorder=2)
            center = _draw_hull(z_pts, color=color, alpha=0.12, edge_color="gray", lw=0.8)
            if center is None and z_pts.size:
                center = z_pts.mean(axis=0)
            if center is not None:
                ax.text(center[0], center[1], str(z + 1),
                        ha="center", va="center", fontsize=12, weight="bold", alpha=0.10, zorder=3)

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return ax
