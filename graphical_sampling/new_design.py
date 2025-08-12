from __future__ import annotations
from copy import deepcopy
import numpy as np
from .sampling import KMeansSampler


class NewDesign:
    def __init__(
        self,
        kmeans: KMeansSampler,
    ):
        self.kmeans = kmeans
        self.rng = np.random.default_rng()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NewDesign):
            return False
        return self.kmeans.clusters == other.kmeans.clusters

    def copy(self) -> NewDesign:
        new_kmeans = KMeansSampler(
            population = self.kmeans.population,
            n=self.kmeans.n,
            n_zones=self.kmeans.n_zones_value,
            split_size=self.kmeans.split_size,
            zone_builder=self.kmeans.zone_builder_str,
            units_order='Change',
            zones_order='Change',
            clusters=deepcopy(self.kmeans.clusters),
            labels=self.kmeans.labels,
            centroids=self.kmeans.centroids,
        )
        return NewDesign(new_kmeans)

    def iterate(
            self,
            n_clusters_to_change_order_zone: int | str = 'all',
            n_clusters_to_change_order_units: int | str = 'all',
            n_zones_to_change_order_units: int | str = 'all',
            n_changes_in_order_of_units: int = 1,
            n_changes_in_order_of_zones: int = 1,
    ) -> None:
        self.kmeans.reorder_change(
            n_clusters_to_change_order_zone,
            n_clusters_to_change_order_units,
            n_zones_to_change_order_units,
            n_changes_in_order_of_units,
            n_changes_in_order_of_zones,
        )

    def __hash__(self) -> int:
        return hash(tuple(self.kmeans.clusters))
