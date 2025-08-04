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
    """
    Plot a population and its clusters with convex hulls and zone numbering.

    Args:
        population: A Population instance with attributes coords (Nx2) and probs (N,).
        clusters: List of Cluster instances, each with zones (a list of Zone objects).
        connect_points: If True, draw lines connecting zone points in the order of index_share.
        ax: Matplotlib Axes to draw on. Creates new one if None.
        background_gdf: Optional GeoDataFrame to plot as a background.
        size_scale: Scaling factor for marker sizes (relative to probabilities).
        figsize: Figure size if ax is None.
        dpi: Dots per inch.

    Returns:
        ax: The Matplotlib Axes with the plot.
    """
    # Helper to draw a convex hull polygon if possible
    def _draw_hull(points: np.ndarray, color: str, alpha: float, edge_color: str, lw: float):
        if points.shape[0] < 3:
            return None
        try:
            hull = ConvexHull(points)
            verts = points[hull.vertices]
        except (QhullError, ValueError):
            return None
        poly = Polygon(verts, closed=True, facecolor=color, alpha=alpha,
                       edgecolor=edge_color, lw=lw)
        ax.add_patch(poly)
        return verts.mean(axis=0)

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot background if provided
    if background_gdf is not None:
        background_gdf.plot(ax=ax, color="white", edgecolor="black", linewidth=1.5, zorder=0)

    # Color cycle for clusters
    colors = plt.get_cmap('tab10').colors

    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]

        # Aggregate units in cluster: indices and shares
        idx_share = cluster.get_index_share(reduce=True)
        if idx_share.size == 0:
            continue
        unit_idx = idx_share[:, 0].astype(int)
        shares = idx_share[:, 1]
        coords = population.coords[unit_idx]
        probs = population.probs[unit_idx] * shares

        # Cluster hull
        _draw_hull(coords, color=color, alpha=0.3, edge_color='black', lw=1.0)

        # Scatter units, size ~ probability
        sizes = probs * size_scale
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color=color,
                   edgecolors='none', alpha=0.8, zorder=2)

        # Plot each zone within cluster
        for z, zone in enumerate(cluster.zones):
            zid = z + 1
            zone_idx = zone.index.astype(int)
            z_coords = population.coords[zone_idx]

            # Optionally connect zone points in data order
            if connect_points and z_coords.shape[0] > 1:
                ax.plot(z_coords[:, 0], z_coords[:, 1], linestyle='-',
                        color=color, alpha=0.7, zorder=2)

            # zone hull and compute centroid for label
            center = _draw_hull(z_coords, color=color, alpha=0.15,
                                edge_color='gray', lw=0.8)
            if center is None:
                # fallback to mean of points
                center = z_coords.mean(axis=0) if z_coords.size else None
            if center is not None:
                ax.text(center[0], center[1], str(zid), ha='center', va='center',
                        fontsize=12, weight='bold', alpha=0.5, zorder=3)

    # Clean up axes
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return ax
