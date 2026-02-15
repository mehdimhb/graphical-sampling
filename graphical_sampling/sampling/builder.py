import numpy as np
from typing import List, Tuple, Optional
from itertools import pairwise
from abc import ABC, abstractmethod
from copy import copy

# Corrected imports from your package structure
from ..clustering import FIPBalancedNMeans
from .entity import Population, Zone, Cluster


class BaseZoneBuilder(ABC):
    """
    Abstract Base Class for building Zone objects from a subset of a Population.
    Defines common methods and an interface for different zone generation strategies.
    """

    def __init__(self):
        self._zone_id_counter = 0  # To assign unique IDs to zones

    def _get_next_zone_id(self) -> int:
        zone_id = self._zone_id_counter
        self._zone_id_counter += 1
        return zone_id

    def _numerical_stabilizer(self, population: Population, index_share: np.ndarray) -> np.ndarray:
        """
        Stabilizes numerical probabilities by rounding and re-normalizing.
        """
        index = index_share[:, 0].astype(np.int64)
        share = index_share[:, 1]
        probs_stabled = np.round(share * population.probs[index], 9)
        probs_stabled[probs_stabled < 1e-9] = 0.0

        original_sum = np.sum(share)
        stabilized_sum = np.sum(probs_stabled)

        if original_sum > 1e-9 and stabilized_sum > 1e-9:
            probs_stabled = probs_stabled * (original_sum / stabilized_sum)

        return np.column_stack((index, probs_stabled))

    @abstractmethod
    def build_zones(self, population: Population, index_share: np.ndarray) -> List[Zone]:
        pass


class ClusteringZoneBuilder(BaseZoneBuilder):
    """
    Builds Zone objects by applying FIPBalancedNMeans clustering.
    """

    def __init__(self, n_zones: int):
        super().__init__()
        if not isinstance(n_zones, int) or n_zones <= 0:
            raise ValueError("n_zones must be a positive integer.")
        self._n_zones = n_zones

    def build_zones(self, population: Population, index_share: np.ndarray) -> List[Zone]:
        self._zone_id_counter = 0 

        if index_share.size == 0:
            return []

        if self._n_zones == 1:
            return [Zone(id=self._get_next_zone_id(), _pop=population, _index_share=index_share)]

        indices = index_share[:, 0].astype(np.int64)
        shares = index_share[:, 1]

        # Use FIPBalancedNMeans (No split_size)
        upb_kmeans = FIPBalancedNMeans(n=self._n_zones)
        upb_kmeans.fit(population.subset(indices, shares))

        zones: List[Zone] = []
        for zone_cluster in upb_kmeans.clusters:
            free_indices = zone_cluster['free']
            b = zone_cluster['border']
            
            # Map FIP border structure to Zone index_share
            z_index_share = np.column_stack((free_indices, np.ones(free_indices.size)))
            
            if b['floor_index'] != -1:
                floor = np.array([b['floor_index'], b['floor_percentage']], dtype=float)
                z_index_share = np.vstack([floor, z_index_share])
            
            if b['ceil_index'] != -1:
                ceil = np.array([b['ceil_index'], b['ceil_percentage']], dtype=float)
                z_index_share = np.vstack([z_index_share, ceil])

            zones.append(Zone(id=self._get_next_zone_id(), _pop=population, _index_share=z_index_share))

        return zones


class SweepingZoneBuilder(BaseZoneBuilder):
    """
    Builds Zone objects using a two-pass sweep-line algorithm.
    """

    def __init__(self, n_zones: Tuple[int, int]):
        super().__init__()
        if not isinstance(n_zones, tuple) or len(n_zones) != 2 or \
           not all(isinstance(n, int) and n > 0 for n in n_zones):
            raise ValueError("n_zones must be a tuple (rows, cols) of positive integers.")
        self._n_zones = n_zones

    def build_zones(self, population: Population, index_share: np.ndarray) -> List[Zone]:
        self._zone_id_counter = 0 
        if index_share.size == 0:
            return []

        num_rows, num_cols = self._n_zones
        total_target_zones = num_rows * num_cols
        index = index_share[:, 0].astype(np.int64)

        sort_order_x = np.argsort(population.x[index])
        sorted_by_x = index_share[sort_order_x]

        total_prob = population.sum_prob(sorted_by_x[:, 0], sorted_by_x[:, 1])
        if total_prob == 0:
            return []

        target_prob_per_zone = total_prob / total_target_zones
        x_sweep_threshold = target_prob_per_zone * num_cols

        vertical_segments = self._sweep(population, sorted_by_x, x_sweep_threshold)

        all_zones: List[Zone] = []
        for v_seg in vertical_segments:
            if v_seg.size == 0: continue
            
            v_indices = v_seg[:, 0].astype(np.int64)
            sort_order_y = np.argsort(population.y[v_indices])
            sorted_by_y = v_seg[sort_order_y]

            horizontal_segments = self._sweep(population, sorted_by_y, target_prob_per_zone)

            for h_seg in horizontal_segments:
                if h_seg.size == 0: continue
                all_zones.append(Zone(id=self._get_next_zone_id(), _pop=population, _index_share=h_seg))

        return all_zones

    def _sweep(self, population: Population, sorted_is: np.ndarray, threshold: float) -> List[np.ndarray]:
        index = sorted_is[:, 0].astype(np.int64)
        share = sorted_is[:, 1]
        cumulative_probs = np.cumsum(population.probs[index] * share)
        total_probs = cumulative_probs[-1] if cumulative_probs.size > 0 else 0.0

        if total_probs < threshold - 1e-9:
            return [sorted_is]

        thresholds = np.arange(threshold, total_probs - 1e-9, threshold)
        indices_at_thresholds = np.searchsorted(cumulative_probs, thresholds, side="right")
        split_points = np.concatenate(([0], indices_at_thresholds, [sorted_is.shape[0]]))

        segments = []
        for i, j in pairwise(split_points):
            segments.append(sorted_is[i:j])
        return segments


class ClusterBuilder:
    """
    Divides Population into clusters and then applies a ZoneBuilder to each.
    """

    def __init__(self, n_clusters: int, zone_builder: BaseZoneBuilder):
        self.n_clusters = n_clusters  # Required for your KMeansSampler call
        self._zone_builder = zone_builder
        self._next_cluster_id = 0 

        if not isinstance(self.n_clusters, int) or self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")

    def _get_next_cluster_id(self) -> int:
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        return cluster_id

    def build_clusters(self, population: Population) -> Tuple[List[Cluster], np.ndarray, np.ndarray]:
        self._next_cluster_id = 0 
        
        # Fit FIPBalancedNMeans (No split_size)
        upb_kmeans = FIPBalancedNMeans(n=self.n_clusters)
        upb_kmeans.fit(population)

        clusters: List[Cluster] = []
        for cluster_info in upb_kmeans.clusters:
            free_indices = cluster_info['free']
            index_share = np.column_stack((free_indices, np.ones(free_indices.size)))

            zones_for_cluster = self._zone_builder.build_zones(population, index_share)
            
            b = cluster_info['border']
            floor = np.array([b['floor_index'], b['floor_percentage']], dtype=float) if b['floor_index'] != -1 else None
            ceil = np.array([b['ceil_index'], b['ceil_percentage']], dtype=float) if b['ceil_index'] != -1 else None

            new_cluster = Cluster(
                id=self._get_next_cluster_id(),
                _pop=population,
                _zones=zones_for_cluster,
                _floor=floor,
                _ceil=ceil,
            )
            clusters.append(new_cluster)

        return clusters, upb_kmeans.labels, upb_kmeans.centroids


class NestedClusterBuilder:
    """
    Builds clusters hierarchically.
    """

    def __init__(self, n_clusters: tuple[int], zone_builder: BaseZoneBuilder):
        self._n_clusters = n_clusters
        self._zone_builder = zone_builder
        self._next_cluster_id = 0 

    def _get_next_cluster_id(self) -> int:
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        return cluster_id

    def build_clusters(self, population: Population) -> Tuple[List[Cluster], np.ndarray, np.ndarray]:
        self._next_cluster_id = 0 
        
        # Initial raw cluster containing full population
        clusters_raw = [{
            'free': np.arange(population.N),
            'border': {'floor_index': -1, 'ceil_index': -1, 'floor_percentage': 0.0, 'ceil_percentage': 0.0}
        }]
        
        i = 0
        while len(clusters_raw) < np.prod(self._n_clusters):
            new_clusters = []
            for c in clusters_raw:
                # Sub-clustering logic
                fbn = FIPBalancedNMeans(n=self._n_clusters[i])
                indices = c['free']
                fbn.fit(population.subset(indices))
                new_clusters.extend(fbn.clusters)
            clusters_raw = new_clusters
            i += 1

        # Convert raw dicts to Cluster objects
        clusters: List[Cluster] = []
        membership = np.zeros((population.N, len(clusters_raw)))
        
        for idx, c_info in enumerate(clusters_raw):
            free = c_info['free']
            z_is = np.column_stack((free, np.ones(free.size)))
            
            zones = self._zone_builder.build_zones(population, z_is)
            
            b = c_info['border']
            floor = np.array([b['floor_index'], b['floor_percentage']], dtype=float) if b['floor_index'] != -1 else None
            ceil = np.array([b['ceil_index'], b['ceil_percentage']], dtype=float) if b['ceil_index'] != -1 else None

            clusters.append(Cluster(
                id=self._get_next_cluster_id(),
                _pop=population,
                _zones=zones,
                _floor=floor,
                _ceil=ceil
            ))
            
            membership[free, idx] = 1.0
            if b['floor_index'] != -1: membership[b['floor_index'], idx] = b['floor_percentage']
            if b['ceil_index'] != -1: membership[b['ceil_index'], idx] = b['ceil_percentage']

        labels = np.argmax(membership, axis=1)
        centroids = np.array([population.coords[labels == k].mean(axis=0) if np.any(labels == k) 
                              else np.zeros(2) for k in range(len(clusters))])

        return clusters, labels, centroids