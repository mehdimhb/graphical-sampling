from functools import cached_property
from math import sqrt
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Union, Optional

# Import the new builder classes and entities from their expected relative paths
from ..structs import Sample
from ..population import Population
from ..clustering import AuxiliaryBalancedKMeans
from .entity import Zone, Cluster  # Assuming these are in .entity
from .builder import ClusteringZoneBuilder, SweepingZoneBuilder, ClusterBuilder, \
    BaseZoneBuilder, ClusterInClusterBuilder # Assuming these are in .builder
from .order import Order, LexicoXY, LexicoYX, Random, Angle, DistFromOrigin, Projection, DistFromCentroid, \
    Spiral, MaxCoord, Snake, HilbertCurve, Change  # All OrderStrategy implementations
from ..design import Design
from ..measure import Density, Moran, LocalBalance, Voronoi


class KMeansSampler:
    def __init__(
            self,
            population: Population,  # Now takes a Population object directly
            *,
            n: int | tuple[int],  # This n is likely n_clusters for Sampler
            n_clusters: int,
            n_zones: int | Tuple[int, int],  # Can be int for clustering, tuple for sweep
            split_size: float,
            cluster_in_cluster: bool = False,
            zone_builder: str = "sweep",
            units_order: str = "lexico-xy",  # Default for units within zones
            zones_order: str = "lexico-xy",  # Default for zones within clusters
            clusters: List[Cluster] = None,
            labels = None,
            centroids = None,
    ) -> None:
        if not isinstance(population, Population):
            raise TypeError("Input 'population' must be an instance of the Population class.")

        if n_clusters > n:
            raise ValueError(f"Number of clusters ({n_clusters}) cannot exceed n ({n}).")
        if n % n_clusters != 0:
            raise ValueError(f"n/num_clusters must be an integer, got n={n}, clusters={n_clusters}.")

        # Store the Population object
        self.population = population
        self.coords = self.population.coords
        self.probs = self.population.probs

        self.n = np.prod(n) if cluster_in_cluster else n  # Number of clusters
        self.n_clusters = n_clusters
        self.cluster_in_cluster = cluster_in_cluster
        self.split_size = split_size
        self.zone_builder_str = zone_builder

        # Store the current sorting strategies (as strings)
        self._current_units_order_str = units_order
        self._current_zones_order_str = zones_order

        # 1. Convert string sort methods to OrderStrategy instances for initial build
        units_order_strategy = self._get_order_strategy(self._current_units_order_str)
        zones_order_strategy = self._get_order_strategy(self._current_zones_order_str)

        # 2. Initialize ZoneBuilder based on zone_mode
        zone_builder_obj: BaseZoneBuilder
        if self.zone_builder_str == "cluster":
            if not isinstance(n_zones, int):
                raise ValueError("n_zones must be an integer for 'cluster' zone_mode.")
            self.n_zones_value = n_zones  # Store the actual value for later use
            zone_builder_obj = ClusteringZoneBuilder(
                n_zones=self.n_zones_value,
                split_size=self.split_size
            )
        elif self.zone_builder_str == "sweep":
            if isinstance(n_zones, int):
                self.n_zones_value = (n_zones, n_zones)  # Convert int to tuple (n, n) for sweep
            elif isinstance(n_zones, tuple) and len(n_zones) == 2:
                self.n_zones_value = n_zones
            else:
                raise ValueError("n_zones must be an integer or tuple (rows, cols) for 'sweep' zone_mode.")
            zone_builder_obj = SweepingZoneBuilder(
                n_zones=self.n_zones_value
            )
        else:
            raise ValueError(f"Invalid zone_mode: {self.zone_builder_str}. Must be 'cluster' or 'sweep'.")

        # 3. Initialize ClusterBuilder
        if cluster_in_cluster:
            self.cluster_builder = ClusterInClusterBuilder(
                n_clusters=n,
                zone_builder=zone_builder_obj,
                split_size=self.split_size
            )
        else:
            self.cluster_builder = ClusterBuilder(
                # n_clusters=self.n,
                n_clusters=self.n_clusters,
                zone_builder=zone_builder_obj,
                split_size=self.split_size
            )

        # 4. Build clusters (which internally build zones)
        if clusters is None:
            self.clusters, self.labels, self.centroids = self.cluster_builder.build_clusters(self.population)
            self.reorder()
        else:
            self.clusters = clusters
            self.labels = labels
            self.centroids = centroids

        self.rng = np.random.default_rng()

        # Eagerly compute cached properties on first access (handled by @cached_property)
        # We trigger access here to ensure they are available immediately after init.
        _ = self.design
        _ = self.all_samples
        _ = self.all_samples_probs
        _ = self.density

        # Ensure total probability sums to 1 after potential filtering and initial build
        # if np.sum(self.all_samples_probs) > 0:
        #     self.all_samples_probs *= 1 / np.sum(self.all_samples_probs)

    def _get_order_strategy(self, method_name: str) -> Order:
        """Returns an instance of an OrderStrategy based on the method name."""
        strategy_map = {
            "lexico-yx": LexicoYX(),
            "lexico-xy": LexicoXY(),
            "random": Random(),
            "angle_0": Angle(),
            "distance_0": DistFromOrigin(),
            "projection": Projection(),
            "center": DistFromCentroid(),
            "spiral": Spiral(),
            "max": MaxCoord(),
            "snake": Snake(),
            "hilbert": HilbertCurve(min_coord=np.array([0.,0.]), max_coord=np.array([1.,1.])),
            "change": Change(1),
        }

        strategy = strategy_map.get(method_name.lower())
        if strategy is None:
            raise ValueError(f"Unknown sort method: {method_name}. Available: {list(strategy_map.keys())}")
        return strategy

    def reorder(
            self,
            new_units_order: Optional[str] = None,
            new_zones_order: Optional[str] = None
    ) -> None:
        """
        Reorders the units within zones, zones within clusters, and optionally clusters themselves.
        Invalidates and recomputes the sampling design after reordering.

        Args:
            new_units_order (str, optional): The new sorting method for units within zones.
                                                  If None, retains current method.
            new_zones_order (str, optional): The new sorting method for zones within clusters.
                                                   If None, retains current method.
        """
        units_order_to_apply = self._get_order_strategy(
            new_units_order) if new_units_order else self._get_order_strategy(
            self._current_units_order_str)
        zones_order_to_apply = self._get_order_strategy(
            new_zones_order) if new_zones_order else self._get_order_strategy(
            self._current_zones_order_str)

        # Update current method strings
        if new_units_order: self._current_units_order_str = new_units_order
        if new_zones_order: self._current_zones_order_str = new_zones_order

        # 1. Reorder units within zones and zones within clusters
        for cluster in self.clusters:
            # Reorder units within each zone
            for zone in cluster.zones:
                zone.apply_order(units_order_to_apply)

            # Reorder zones within the cluster
            cluster.apply_order(zones_order_to_apply)

        # 3. Invalidate cached properties to force re-computation
        if 'design' in self.__dict__: del self.design
        if 'all_samples' in self.__dict__: del self.all_samples
        if 'all_samples_probs' in self.__dict__: del self.all_samples_probs
        if 'density_scores' in self.__dict__: del self.density_scores
        if 'moran_scores' in self.__dict__: del self.moran_scores
        if 'local_balance_scores' in self.__dict__: del self.local_balance_scores
        if 'voronoi_scores' in self.__dict__: del self.voronoi_scores


        # Re-compute relevant properties
        _ = self.design
        _ = self.all_samples
        _ = self.all_samples_probs
        _ = self.density_scores
        _ = self.moran_scores
        _ = self.local_balance_scores
        _ = self.voronoi_scores

    def reorder_change(
            self,
            n_clusters_to_change_order_zone: int | str = 'all',
            n_clusters_to_change_order_units: int | str = 'all',
            n_zones_to_change_order_units: int | str = 'all',
            n_changes_in_order_of_units: int = 1,
            n_changes_in_order_of_zones: int = 1,
    ) -> None:
        units_order_to_apply = Change(num_changes=n_changes_in_order_of_units)
        zones_order_to_apply = Change(num_changes=n_changes_in_order_of_zones)

        if n_clusters_to_change_order_zone == 'all':
            for cluster in self.clusters:
                cluster.apply_order(zones_order_to_apply)
        else:
            # print('REORDERING ZONES')
            clusters_to_change = np.random.choice(np.arange(len(self.clusters)), size=n_clusters_to_change_order_zone, replace=False)
            # print(clusters_to_change)
            for i in clusters_to_change:
                # print('BEFORE', self.clusters[i])
                self.clusters[i].apply_order(zones_order_to_apply)
                # print('AFTER', self.clusters[i])


        if n_clusters_to_change_order_units == 'all':
            for cluster in self.clusters:
                if n_zones_to_change_order_units == 'all':
                    for zone in cluster.zones:
                        zone.apply_order(units_order_to_apply)
                else:
                    zones_to_change = np.random.choice(np.arange(len(cluster)), size=n_zones_to_change_order_units, replace=False)
                    for i in zones_to_change:
                        cluster.zones[i].apply_order(units_order_to_apply)
        else:
            # print('REORDERING UNITS')
            clusters_to_change = np.random.choice(np.arange(len(self.clusters)), size=n_clusters_to_change_order_units, replace=False)
            # print(clusters_to_change)
            for i in clusters_to_change:
                cluster = self.clusters[i]
                if n_zones_to_change_order_units == 'all':
                    for zone in cluster.zones:
                        zone.apply_order(units_order_to_apply)
                else:
                    zones_to_change = np.random.choice(np.arange(len(cluster)), size=n_zones_to_change_order_units, replace=False)
                    # print(zones_to_change)
                    for j in zones_to_change:
                        # print('BEFORE', cluster.zones[j])
                        # print('BEFORE', cluster.zones[0])
                        cluster.zones[j].apply_order(units_order_to_apply)
                        # print('AFTER', cluster.zones[j])
                        # print('AFTER', cluster.zones[0])


        # 3. Invalidate cached properties to force re-computation
        if 'design' in self.__dict__: del self.design
        if 'all_samples' in self.__dict__: del self.all_samples
        if 'all_samples_probs' in self.__dict__: del self.all_samples_probs
        if 'density_scores' in self.__dict__: del self.density_scores
        if 'moran_scores' in self.__dict__: del self.moran_scores
        if 'local_balance_scores' in self.__dict__: del self.local_balance_scores


        # Re-compute relevant properties
        _ = self.design
        _ = self.all_samples
        _ = self.all_samples_probs
        _ = self.density_scores
        _ = self.moran_scores
        _ = self.local_balance_scores

    def sample(self, n_samples: int):
        if not self.clusters:
            return np.zeros((n_samples, self.n), dtype=int)

        m = self.n // self.n_clusters  # picks per cluster

        # Determine conceptual zones as before (used for clamping)
        if self.zone_builder_str == "cluster":
            total_conceptual_zones = self.n_zones_value
        elif self.zone_builder_str == "sweep":
            total_conceptual_zones = self.n_zones_value[0] * self.n_zones_value[1]
        else:
            total_conceptual_zones = 0

        if total_conceptual_zones == 0:
            total_conceptual_zones = max(
                1,
                len(self.clusters[0].zones) if self.clusters and self.clusters[0].zones else 1
            )

        # We now return n picks per sample (m per cluster)
        samples = np.zeros((n_samples, self.n), dtype=int)

        for i in range(n_samples):
            r = self.rng.random()  # base in [0,1)

            # Use only the first self.n_clusters clusters
            for j in range(self.n_clusters):
                cluster = self.clusters[j]
                col_start = j * m
                col_end = col_start + m

                if not cluster.zones or getattr(cluster, "zones_edges", None) is None or len(cluster.zones_edges) == 0:
                    samples[i, col_start:col_end] = -1
                    continue

                for k in range(m):
                    rk = r + k  # r, r+1, ..., r+(m-1)  (edges already in [0, m))

                    # ---- Zone selection on existing edges (already [0, m)) ----
                    zone_index = np.searchsorted(cluster.zones_edges, rk, side="right") - 1
                    zone_index = min(zone_index, total_conceptual_zones - 1)
                    zone_index = max(0, zone_index)

                    current_zone = cluster.zones[zone_index]
                    unit_probs = np.asarray(current_zone.prob, dtype=float)

                    if unit_probs.size == 0 or np.sum(unit_probs) < 1e-9:
                        samples[i, col_start + k] = -1
                        continue

                    zone_start_global_prob = cluster.zones_edges[zone_index]

                    # Per-zone cumulative already aligned to global [0, m)
                    global_cumulative_unit_probs = np.cumsum(unit_probs) + zone_start_global_prob

                    # ---- Unit selection ----
                    unit_selection_idx = np.searchsorted(global_cumulative_unit_probs, rk, side="right")
                    unit_selection_idx = min(unit_selection_idx, len(current_zone) - 1)
                    unit_selection_idx = max(0, unit_selection_idx)

                    if len(current_zone) > 0:
                        sel_idx0 = current_zone.index[unit_selection_idx]
                        # samples[i, col_start + k] = self.population.ids[sel_idx0]
                        samples[i, col_start + k] = sel_idx0
                    else:
                        samples[i, col_start + k] = -1

        return samples

    @cached_property
    # @property
    def design(self) -> Design:
        # --- Conceptual zones (for clamping) ---
        if self.zone_builder_str == "cluster":
            total_conceptual_zones = self.n_zones_value
        elif self.zone_builder_str == "sweep":
            total_conceptual_zones = self.n_zones_value[0] * self.n_zones_value[1]
        else:
            raise ValueError("Invalid zone_mode for _build_design.")

        if total_conceptual_zones == 0:
            return Design(inclusions=None)

        # --- Validate n and number of clusters ---
        if not getattr(self, "n_clusters", 0):
            return Design(inclusions=None)

        num_clusters = int(self.n_clusters)
        n = int(self.n)

        m = n // num_clusters  # number of offsets per cluster

        # --- Collect fractional cut points on [0,1) ---
        # We take all global boundaries in [0, m), reduce modulo 1, and deduplicate with rounding.
        cuts_frac = set()

        for cluster in self.clusters:
            if getattr(cluster, "zones_edges", None) is None or not cluster.zones:
                continue

            # zone boundary cuts (mod 1)
            for edge in cluster.zones_edges:
                frac = edge - np.floor(edge)
                cuts_frac.add(round(float(frac), 12))

            # within-zone unit boundary cuts (mod 1)
            for zone_idx, zone in enumerate(cluster.zones):
                if len(zone) == 0:
                    continue
                unit_probs = np.asarray(zone.prob, dtype=float)
                if unit_probs.size == 0:
                    continue

                base = float(cluster.zones_edges[zone_idx])  # zone start (global, [0,m))
                cum = np.cumsum(unit_probs)  # within-zone cumulative
                globals_ = base + cum  # global boundaries in [0,m)
                for b in globals_:
                    frac = b - np.floor(b)
                    cuts_frac.add(round(float(frac), 12))

        # Build sorted partition on [0,1]
        pts = sorted(c for c in cuts_frac if 0.0 <= c <= 1.0)
        if not pts:
            return Design(inclusions=None)
        if pts[0] != 0.0:
            pts.insert(0, 0.0)
        if pts[-1] != 1.0:
            pts.append(1.0)

        design = Design(inclusions=None)

        # --- For each interval, compute the full n-sized sample via r, r+1, ..., r+(m-1) ---
        last = pts[0]
        for p in pts[1:]:
            length = round(p - last, 12)  # probability weight for r in this interval
            if length <= 0:
                last = p
                continue

            mid = last + (p - last) / 2.0  # r in [0,1)

            ids_in_sample = []
            for cluster in self.clusters:
                if getattr(cluster, "zones_edges", None) is None or not cluster.zones:
                    continue

                # For each offset k, select a unit for (mid + k)
                for k in range(m):
                    rk = mid + k  # lives in [k, k+1)

                    # --- Zone selection on [0, m) ---
                    zone_index = np.searchsorted(cluster.zones_edges, rk, side="right") - 1
                    # clamp by conceptual zone count and the actual cluster's zone count
                    zone_index = max(0, min(zone_index, total_conceptual_zones - 1,
                                            len(cluster.zones) - 1))

                    current_zone = cluster.zones[zone_index]
                    unit_probs = np.asarray(current_zone.prob, dtype=float)
                    if unit_probs.size == 0:
                        continue

                    zone_start = float(cluster.zones_edges[zone_index])
                    global_cum = np.cumsum(unit_probs) + zone_start  # global in [0,m)

                    unit_selection_idx = np.searchsorted(global_cum, rk, side="right")
                    unit_selection_idx = max(0, min(unit_selection_idx, len(current_zone) - 1))

                    if current_zone.index.size > 0:
                        sel0 = int(current_zone.index[unit_selection_idx])  # 0-based pop index
                        ids_in_sample.append(sel0)

            if ids_in_sample:
                design.push(Sample(length, frozenset(ids_in_sample)))

            last = p

        design.merge_identical()
        return design

    @cached_property
    def density(self) -> Density:
        return Density(
            self.population,
            self.n,
            self.split_size,
            self.labels,
            self.centroids
        )

    @cached_property
    def all_samples(self) -> NDArray:
        samples_list = []
        for sample_obj in self.design:
            if len(sample_obj.ids) == self.n and sample_obj.probability > 1e-9:
                samples_list.append(list(sample_obj.ids))

        if not samples_list:
            return np.array([]).reshape(0, self.n)
        return np.array(samples_list)

    @cached_property
    def all_samples_probs(self) -> NDArray:
        samples_probs_list = []
        for sample_obj in self.design:
            if len(sample_obj.ids) == self.n and sample_obj.probability > 1e-9:
                samples_probs_list.append(sample_obj.probability)

        if not samples_probs_list:
            return np.array([])
        return np.array(samples_probs_list)

    @cached_property
    def fips(self) -> NDArray:
        fip = np.zeros_like(self.probs)
        for sample, prob in zip(self.all_samples, self.all_samples_probs):
            fip[sample] += prob
        return np.array(fip)

    @cached_property
    def density_scores(self):
        if self.all_samples.size == 0:
            return np.array([])
        return self.density.score(self.all_samples)

    @cached_property
    def moran_scores(self):
        if self.all_samples.size == 0:
            return np.array([])
        moran = Moran(self.population)
        return moran.score(self.all_samples)

    @cached_property
    def local_balance_scores(self):
        if self.all_samples.size == 0:
            return np.array([])
        local_balance = LocalBalance(self.population)
        return local_balance.score(self.all_samples)

    @cached_property
    def voronoi_scores(self):
        if self.all_samples.size == 0:
            return np.array([])
        voronoi = Voronoi(self.population)
        return voronoi.score(self.all_samples)

    def expected_score(self, all_samples_scores: NDArray) -> float:
        if all_samples_scores.size == 0 or self.all_samples_probs.size == 0:
            ValueError("There are no samples to compute density scores for.")

        return float(np.sum(all_samples_scores * self.all_samples_probs))

    def expected_density_score(self) -> float:
        return self.expected_score(self.density_scores)

    def expected_moran_score(self) -> float:
        return self.expected_score(self.moran_scores)

    def expected_local_balance_score(self) -> float:
        return self.expected_score(self.local_balance_scores)

    def expected_voronoi_score(self) -> float:
        return self.expected_score(self.voronoi_scores)

    def var_score(self, all_samples_scores: NDArray) -> float:
        if all_samples_scores.size == 0 or self.all_samples_probs.size == 0:
            ValueError("There are no samples to compute density scores for.")

        expected_val = self.expected_score(all_samples_scores)
        return float(np.sum(all_samples_scores ** 2 * self.all_samples_probs) - expected_val ** 2)

    def var_density_score(self) -> float:
        return self.var_score(self.density_scores)

    def var_moran_score(self) -> float:
        return self.var_score(self.moran_scores)

    def var_local_balance_score(self) -> float:
        return self.var_score(self.local_balance_scores)

    def var_voronoi_score(self) -> float:
        return self.var_score(self.voronoi_scores)

    def std_density_score(self) -> float:
        return sqrt(self.var_score(self.density_scores))

    def std_moran_score(self) -> float:
        return sqrt(self.var_score(self.moran_scores))

    def std_local_balance_score(self) -> float:
        return sqrt(self.var_score(self.local_balance_scores))

    def std_voronoi_score(self) -> float:
        return sqrt(self.var_score(self.voronoi_scores))

    def score_summary_df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame summarizing expected score and standard deviation
        for the four measures: density, moran, local_balance, and voronoi.

        Index:   measure name
        Columns: ["expected", "std"]
        """
        data = {
            "measure": ["density", "moran", "local_balance", "voronoi"],
            "expected": [
                self.expected_density_score(),
                self.expected_moran_score(),
                self.expected_local_balance_score(),
                self.expected_voronoi_score(),
            ],
            "std": [
                self.std_density_score(),
                self.std_moran_score(),
                self.std_local_balance_score(),
                self.std_voronoi_score(),
            ],
        }
        return pd.DataFrame(data).set_index("measure")