from functools import cached_property
from math import sqrt
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Union, Optional

# Import the builder classes and entities from their expected relative paths
from .design import Design
from .population import Population
from .clustering import FIPBalancedNMeans
from .entity import Zone, Cluster  
from .builder import ClusteringZoneBuilder, SweepingZoneBuilder, ClusterBuilder, \
    BaseZoneBuilder, NestedClusterBuilder 
from .order import Order, LexicoXY, LexicoYX, Random, Angle, DistFromOrigin, Projection, DistFromCentroid, \
    Spiral, MaxCoord, Snake, HilbertCurve, Change  
from .index import Density, Moran, LocalBalance, Voronoi


class KMeansSampler:
    def __init__(
            self,
            population: Population,
            *,
            n: int,  # n_clusters for Sampler
            n_zones: int | Tuple[int, int],
            zone_builder: str = "sweep",
            units_order: str = "lexico-xy",
            zones_order: str = "lexico-xy",
            clusters: List[Cluster] = None,
            labels = None,
            centroids = None,
    ) -> None:
        if not isinstance(population, Population):
            raise TypeError("Input 'population' must be an instance of the Population class.")

        # Store the Population object
        self.population = population
        self.coords = self.population.coords
        self.probs = self.population.probs

        self.n = n
        self.zone_builder_str = zone_builder

        # Store the current sorting strategies
        self._current_units_order_str = units_order
        self._current_zones_order_str = zones_order

        # 1. Initialize ZoneBuilder based on zone_mode (split_size removed)
        zone_builder_obj: BaseZoneBuilder
        if self.zone_builder_str == "cluster":
            if not isinstance(n_zones, int):
                raise ValueError("n_zones must be an integer for 'cluster' zone_mode.")
            self.n_zones_value = n_zones
            zone_builder_obj = ClusteringZoneBuilder(
                n_zones=self.n_zones_value
            )
        elif self.zone_builder_str == "sweep":
            if isinstance(n_zones, int):
                self.n_zones_value = (n_zones, n_zones)
            elif isinstance(n_zones, tuple) and len(n_zones) == 2:
                self.n_zones_value = n_zones
            else:
                raise ValueError("n_zones must be an integer or tuple (rows, cols) for 'sweep' zone_mode.")
            zone_builder_obj = SweepingZoneBuilder(
                n_zones=self.n_zones_value
            )
        else:
            raise ValueError(f"Invalid zone_mode: {self.zone_builder_str}. Must be 'cluster' or 'sweep'.")

        # 2. Initialize ClusterBuilder (split_size removed)
        self.cluster_builder = ClusterBuilder(
            n_clusters=self.n,
            zone_builder=zone_builder_obj
        )

        # 3. Build clusters (which internally build zones)
        if clusters is None:
            self.clusters, self.labels, self.centroids = self.cluster_builder.build_clusters(self.population)
            self.reorder()
        else:
            self.clusters = clusters
            self.labels = labels
            self.centroids = centroids

        self.rng = np.random.default_rng()

        # Eagerly compute cached properties
        _ = self.design
        _ = self.all_samples
        _ = self.all_samples_probs

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
            raise ValueError(f"Unknown sort method: {method_name}.")
        return strategy

    def reorder(
            self,
            new_units_order: Optional[str] = None,
            new_zones_order: Optional[str] = None
    ) -> None:
        """Reorders the units and zones and invalidates cached properties."""
        units_order_to_apply = self._get_order_strategy(
            new_units_order) if new_units_order else self._get_order_strategy(
            self._current_units_order_str)
        zones_order_to_apply = self._get_order_strategy(
            new_zones_order) if new_zones_order else self._get_order_strategy(
            self._current_zones_order_str)

        if new_units_order: self._current_units_order_str = new_units_order
        if new_zones_order: self._current_zones_order_str = new_zones_order

        for cluster in self.clusters:
            for zone in cluster.zones:
                zone.apply_order(units_order_to_apply)
            cluster.apply_order(zones_order_to_apply)

        self._invalidate_cache()

    def _invalidate_cache(self):
        """Clears all cached properties."""
        cache_attrs = [
            'design', 'all_samples', 'all_samples_probs', 'density',
            'density_scores', 'moran_scores', 'local_balance_scores', 'voronoi_scores'
        ]
        for attr in cache_attrs:
            if attr in self.__dict__:
                del self.__dict__[attr]

    def sample(self, n_samples: int):
        """Standard random sampling based on cumulative probabilities."""
        samples = np.zeros((n_samples, self.n), dtype=int)
        if not self.clusters:
            return samples

        for i in range(n_samples):
            random_number = self.rng.random()
            for j in range(self.n):
                cluster = self.clusters[j]
                cluster_index_share = cluster.get_index_share(reduce=False)
                cluster_index = cluster_index_share[:, 0].astype(int)
                cluster_probs = np.cumsum(cluster_index_share[:, 1] * self.population.probs[cluster_index])

                if cluster_probs.size == 0:
                    samples[i, j] = -1
                    continue

                unit_index = np.searchsorted(cluster_probs, random_number, side="left")
                unit_index = max(0, min(unit_index, cluster_probs.size - 1))
                samples[i, j] = cluster_index[unit_index]

        return samples

    @cached_property
    def design(self) -> Design:
        """Constructs the joint sampling design using simultaneous cumulative cuts."""
        cuts = set()
        for cluster in self.clusters:
            idx_share = cluster.get_index_share(reduce=False)
            c_idx = idx_share[:, 0].astype(int)
            c_probs = np.cumsum(idx_share[:, 1] * self.population.probs[c_idx])
            for cut in c_probs:
                cuts.add(cut)

        pts = sorted(c for c in cuts if 0.0 <= c <= 1.0)
        if not pts: return Design(inclusions=None)
        if pts[0] != 0.0: pts.insert(0, 0.0)
        if pts[-1] != 1.0: pts.append(1.0)

        design = Design(inclusions=None)
        last = pts[0]
        for p in pts[1:]:
            length = round(p - last, 12)
            if length <= 0:
                last = p
                continue
            mid = last + (p - last) / 2.0
            ids_in_sample = []
            for cluster in self.clusters:
                idx_share = cluster.get_index_share(reduce=False)
                c_idx = idx_share[:, 0].astype(int)
                c_probs = np.cumsum(idx_share[:, 1] * self.population.probs[c_idx])
                if c_probs.size == 0: continue
                u_idx = np.searchsorted(c_probs, mid, side="left")
                u_idx = max(0, min(u_idx, c_probs.size - 1))
                ids_in_sample.append(c_idx[u_idx])

            if ids_in_sample:
                design.push(Sample(length, frozenset(ids_in_sample)))
            last = p

        design.merge_identical()
        return design

    @cached_property
    def density(self) -> Density:
        """Initializes Density index (split_size removed)."""
        return Density(
            self.population,
            self.n,
            self.labels,
            self.centroids
        )

    @cached_property
    def all_samples(self) -> NDArray:
        samples_list = [list(s.ids) for s in self.design if len(s.ids) == self.n and s.probability > 1e-9]
        return np.array(samples_list) if samples_list else np.array([]).reshape(0, self.n)

    @cached_property
    def all_samples_probs(self) -> NDArray:
        probs_list = [s.probability for s in self.design if len(s.ids) == self.n and s.probability > 1e-9]
        return np.array(probs_list)

    # --- Scoring methods ---
    @cached_property
    def density_scores(self):
        return self.density.score(self.all_samples) if self.all_samples.size > 0 else np.array([])

    @cached_property
    def moran_scores(self):
        return Moran(self.population).score(self.all_samples) if self.all_samples.size > 0 else np.array([])

    @cached_property
    def local_balance_scores(self):
        return LocalBalance(self.population).score(self.all_samples) if self.all_samples.size > 0 else np.array([])

    @cached_property
    def voronoi_scores(self):
        return Voronoi(self.population).score(self.all_samples) if self.all_samples.size > 0 else np.array([])

    def expected_score(self, scores: NDArray) -> float:
        return float(np.sum(scores * self.all_samples_probs)) if scores.size > 0 else 0.0

    def score_summary_df(self) -> pd.DataFrame:
        metrics = ["density", "moran", "local_balance", "voronoi"]
        expected = [
            self.expected_score(self.density_scores),
            self.expected_score(self.moran_scores),
            self.expected_score(self.local_balance_scores),
            self.expected_score(self.voronoi_scores)
        ]
        # Standard deviation calculation
        stds = [
            sqrt(max(0, np.sum(self.density_scores**2 * self.all_samples_probs) - expected[0]**2)),
            sqrt(max(0, np.sum(self.moran_scores**2 * self.all_samples_probs) - expected[1]**2)),
            sqrt(max(0, np.sum(self.local_balance_scores**2 * self.all_samples_probs) - expected[2]**2)),
            sqrt(max(0, np.sum(self.voronoi_scores**2 * self.all_samples_probs) - expected[3]**2))
        ]
        return pd.DataFrame({"expected": expected, "std": stds}, index=metrics)