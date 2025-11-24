import numpy as np
from typing import List, Tuple, Union, Optional
from itertools import pairwise
from abc import ABC, abstractmethod
from copy import copy

# Assuming these are correctly imported from their respective modules
from ..clustering import UPBalancedKMeans
from .entity import Population, Zone, Cluster


class BaseZoneBuilder(ABC):
    """
    Abstract Base Class for building Zone objects from a subset of a Population.
    Defines common methods and an interface for different zone generation strategies.
    """

    def __init__(self):
        """
        Initializes the BaseZoneBuilder.
        """
        self._zone_id_counter = 0  # To assign unique IDs to zones, initialized to 0

    def _get_next_zone_id(self) -> int:
        """
        Generates and returns a unique integer ID for a new zone.
        This method ensures each zone created by the builder has a distinct identifier.
        """
        zone_id = self._zone_id_counter
        self._zone_id_counter += 1
        return zone_id

    def _numerical_stabilizer(self, population: Population, index_share: np.ndarray) -> np.ndarray:
        """
        Stabilizes numerical probabilities/shares by rounding to a specified tolerance
        and re-normalizing to preserve the original sum if the sum is significant.
        Very small values are clipped to zero.

        Args:
            share (np.ndarray): An array of probability shares.

        Returns:
            np.ndarray: The numerically stabilized shares.
        """
        index = index_share[:, 0].astype(np.int64)
        share = index_share[:, 1]
        probs_stabled = np.round(share * population.probs[index], 9)
        # Clip very small values to zero based on tolerance
        probs_stabled[probs_stabled < 1e-9] = 0.0

        original_sum = np.sum(share)
        stabilized_sum = np.sum(probs_stabled)

        # Re-normalize if the original sum is significant and the stabilized sum is not zero,
        # to ensure the total sum of shares remains consistent after stabilization.
        if original_sum > 1e-9 and stabilized_sum > 1e-9:
            probs_stabled = probs_stabled * (original_sum / stabilized_sum)

        new_index_share = np.column_stack((index, probs_stabled))

        return new_index_share

    @abstractmethod
    def build_zones(self, population: Population, index_share: np.ndarray) -> List[Zone]:
        """
        Abstract method to be implemented by subclasses for specific zone generation logic.
        This method should take the `index_share` array representing the subset of population
        units and their current shares within that subset, and return a list of Zones.

        Args:
            population (Population): The overall Population object currently being processed.
            index_share (np.ndarray): A 2D NumPy array where the first column contains
                                      the 0-based original population indices of units
                                      and the second column contains their corresponding
                                      shares within the current subset.
        """
        pass


class ClusteringZoneBuilder(BaseZoneBuilder):
    """
    Builds Zone objects by applying balanced k-means clustering to the units
    represented by the input `index_share`.
    """

    def __init__(self,
                 n_zones: int,  # Total number of zones to create by clustering
                 split_size: float = 0.001):
        """
        Initializes the ClusteringZoneBuilder.

        Args:
            n_zones (int): The total number of zones to create by clustering
                                       within the given subset of population units.
            split_size (float): Parameter for AuxiliaryBalancedKMeans, controlling the
                                 size of auxiliary splits.
        """
        super().__init__()
        if not isinstance(n_zones, int) or n_zones <= 0:
            raise ValueError("num_zones_to_create must be a positive integer for ClusteringZoneBuilder.")
        self._n_zones = n_zones
        self._split_size = split_size

    def build_zones(self, population: Population, index_share: np.ndarray) -> List[Zone]:
        """
        Generates zones by applying balanced k-means clustering to the units defined
        by the input `index_share`.

        Args:
            population (Population): The overall Population object currently being processed.
            index_share (np.ndarray): A 2D NumPy array where the first column contains
                                      the 0-based original population indices of units
                                      and the second column contains their current shares
                                      within the subset being zoned.

        Returns:
            List[Zone]: A list of constructed Zone objects based on clustering.
        """

        self._zone_id_counter = 0  # Reset counter for each top-level zone build operation

        if index_share.size == 0:
            return []

        # If only one zone is requested or there are no units to zone_cluster, return a single zone
        if self._n_zones == 1:
            # stabilized_zis = self._numerical_stabilizer(population, index_share)
            zone_id = self._get_next_zone_id()
            return [Zone(id=zone_id, _pop=population, _index_share=index_share)]

        # Extract original population indices and shares for clustering
        indices = index_share[:, 0].astype(np.int64)
        shares = index_share[:, 1]

        upb_kmeans = UPBalancedKMeans(k=self._n_zones, split_size=self._split_size)
        upb_kmeans.fit(population.subset(indices, shares))

        # Prepare the data structure required for building zones,
        # which includes the indices of population units and their membership shares.
        # zones_index_share = generate_index_shares(self._n_zones, upb_kmeans.membership, indices)

        zones: List[Zone] = []

        for zone_cluster in upb_kmeans.clusters:
            free_indices = zone_cluster['free']
            # Skip empty clusters that might result from the clustering process

            floor_index = zone_cluster['border']['floor_index']
            floor_pct = zone_cluster['border']['floor_percentage']
            ceil_index = zone_cluster['border']['ceil_index']
            ceil_pct = zone_cluster['border']['ceil_percentage']
            floor = np.array([floor_index, floor_pct], dtype=float) if floor_index != -1 else None
            ceil = np.array([ceil_index, ceil_pct], dtype=float) if ceil_index != -1 else None

            index_share = np.column_stack((free_indices, np.ones(free_indices.size)))
            if floor_index != -1:
                index_share = np.vstack([floor, index_share])
            if ceil_index != -1:
                index_share = np.vstack([index_share, ceil])

            zone_id = self._get_next_zone_id()

            new_zone = Zone(id=zone_id, _pop=population, _index_share=index_share)

            zones.append(new_zone)

        return zones


class SweepingZoneBuilder(BaseZoneBuilder):
    """
    Builds Zone objects using a sweep-line algorithm, potentially in two passes (horizontal and vertical).
    """

    def __init__(self,
                 n_zones: Tuple[int, int],  # (n_rows, n_cols) for sweep
    ):
        """
        Initializes the SweepingZoneBuilder.

        Args:
            n_zones (Tuple[int, int]): A tuple (num_rows, num_cols) specifying the
                                               grid dimensions for the sweep algorithm.
                                               Each must be a positive integer.
        """
        super().__init__()
        if not isinstance(n_zones, tuple) or len(n_zones) != 2 or \
           not all(isinstance(n, int) and n > 0 for n in n_zones):
            raise ValueError(
                "grid_dimensions must be a tuple (num_rows, num_cols) of positive integers for SweepingZoneBuilder.")
        self._n_zones = n_zones

    def build_zones(self, population: Population, index_share: np.ndarray) -> List[Zone]:
        """
        Generates zones using a two-pass sweep-line algorithm (first along X-axis, then along Y-axis).

        Args:
            population (Population): The overall Population object currently being processed.
            index_share (np.ndarray): A 2D NumPy array where the first column contains
                                      the 0-based original population indices of units
                                      and the second column contains their current shares
                                      within the subset being zoned.

        Returns:
            List[Zone]: A list of constructed Zone objects based on the sweep algorithm.
        """
        self._zone_id_counter = 0  # Reset counter for each top-level zone build operation

        if index_share.size == 0:
            return []

        num_rows, num_cols = self._n_zones
        total_target_zones = num_rows * num_cols

        # Extract original population indices from the input index_share
        index = index_share[:, 0].astype(np.int64)

        # Sort units by X-coordinate for the first sweep
        # We need to sort the input index_share array based on the X-coordinate of the units
        x_coords_for_sorting = population.x[index]
        sort_order_x = np.argsort(x_coords_for_sorting)
        sorted_by_x_index_share = index_share[sort_order_x]

        total_prob = population.sum_prob(sorted_by_x_index_share[:, 0], sorted_by_x_index_share[:, 1])
        if total_prob == 0:
            return []

        # Determine the target share for each individual final zone
        target_prob_per_zone = total_prob / total_target_zones

        # First sweep divides into 'num_rows' (conceptually, vertical strips).
        # Each strip should contain `num_cols` zones' worth of share.
        x_sweep_threshold = target_prob_per_zone * num_cols

        vertical_segments_index_share = self._sweep(
            population, sorted_by_x_index_share, x_sweep_threshold
        )

        all_zones: List[Zone] = []
        for v_seg_index_share in vertical_segments_index_share:
            if v_seg_index_share.size == 0:
                continue

            # Extract original population indices for this vertical segment
            v_seg_pop_indices = v_seg_index_share[:, 0].astype(np.int64)

            # Sort units within this vertical segment by Y-coordinate for the second sweep
            y_coords_for_sorting = population.y[v_seg_pop_indices]
            sort_order_y = np.argsort(y_coords_for_sorting)
            sorted_by_y_in_v_seg_index_share = v_seg_index_share[sort_order_y]

            # Second sweep divides each vertical strip into 'num_cols' (conceptually, horizontal zones).
            # Each of these sub-segments should aim for `target_prob_per_zone`.
            y_sweep_threshold = target_prob_per_zone
            horizontal_segments_index_share = self._sweep(
                population, sorted_by_y_in_v_seg_index_share, y_sweep_threshold
            )

            for h_seg_index_share in horizontal_segments_index_share:
                if h_seg_index_share.size == 0:
                    continue

                shares_for_zone = h_seg_index_share[:, 1]
                # The indices for the zone are in the first column
                pop_indices_for_zone = h_seg_index_share[:, 0].astype(np.int64)

                # stabilized_zis = self._numerical_stabilizer(population, zis)

                zone_id = self._get_next_zone_id()

                new_zone = Zone(id=zone_id, _pop=population, _index_share=h_seg_index_share)

                all_zones.append(new_zone)

        return all_zones

    def _sweep(self, population: Population, sorted_index_share: np.ndarray, threshold: float) -> List[np.ndarray]:
        """
        Performs a sweep-line algorithm to divide units into segments based on
        cumulative share, given an `index_share` array that is already sorted
        along one coordinate axis.

        Args:
            sorted_index_share (np.ndarray): A 2D NumPy array similar to `index_share`,
                                             but where the units are sorted along a
                                             specific spatial coordinate (e.g., X or Y).
                                             The first column is original population indices,
                                             second is share.
            threshold (float): The target cumulative share for each sweep segment.
                                     Segments are created such that their total share is
                                     approximately this threshold.

        Returns:
            List[np.ndarray]: A list of `index_share` arrays, where each array
                              contains the population indices and their share
                              for one swept segment. Returns an empty list if
                              `sorted_index_share` is empty.
        """
        # Extract share from the sorted_index_share
        index = sorted_index_share[:, 0].astype(np.int64)
        share = sorted_index_share[:, 1]
        cumulative_probs = np.cumsum(population.probs[index] * share)
        total_probs = cumulative_probs[-1] if cumulative_probs.size > 0 else 0.0

        # If the total share is less than one threshold, return all units as a single segment
        # Using a small epsilon for floating point comparison
        if total_probs < threshold - 1e-9:
            return [sorted_index_share]

        # Calculate thresholds for splitting
        # Ensure thresholds do not exceed the total share, considering floating point inaccuracies
        thresholds = np.arange(threshold, total_probs + 1e-9, threshold)
        # Remove any thresholds that are effectively equal to the total_probs if they create an empty last segment
        if len(thresholds) > 0 and abs(thresholds[-1] - total_probs) < threshold/2:
            thresholds = thresholds[:-1] # Remove the last one if it's too close to total_probs

        indices_at_thresholds = np.searchsorted(cumulative_probs, thresholds, side="right")

        # Determine split points within the sorted_index_share array
        # Add 0 at the beginning and the total number of units at the end
        split_points = np.concatenate(([0], indices_at_thresholds, [sorted_index_share.shape[0]]))

        split_point_share = {i: share[i] for i in split_points[:-1]}

        swept_segments_index_share = []
        # Iterate through the split points to create segments
        for i, j in pairwise(split_points):
            if j == sorted_index_share.shape[0]:
                segment_index_share = sorted_index_share[i:j].copy()
                segment_index_share[0, 1] = split_point_share[i]
            elif i == j:
                remaining_share = self._get_remaining_share(float(split_point_share[i]), float(population.probs[index[i]]), threshold)
                segment_index_share = sorted_index_share[i].copy().reshape(1, 2)
                segment_index_share[0, 1] = split_point_share[i] - remaining_share
                split_point_share[i] = remaining_share
            else:
                remaining_threshold = copy(threshold)
                remaining_threshold -= population.sum_prob(index[i], split_point_share[i])
                if j > i+1:
                    remaining_threshold -= population.sum_prob(index[i+1:j], share[i+1:j])
                remaining_share_ending = self._get_remaining_share(float(split_point_share[j]), float(population.probs[index[j]]), remaining_threshold)
                segment_index_share = sorted_index_share[i:j+1].copy()
                segment_index_share[0, 1] = split_point_share[i]
                segment_index_share[-1, 1] = split_point_share[j] - remaining_share_ending
                split_point_share[i] = 0
                split_point_share[j] = remaining_share_ending

            swept_segments_index_share.append(segment_index_share)

        return swept_segments_index_share

    def _get_remaining_share(self, share: float, prob: float, remaining_threshold: float) -> float:
        current_prob = share * prob
        if remaining_threshold < 1e-9:
            return share
        if current_prob > remaining_threshold + 1e-9:
            return (current_prob - remaining_threshold)/prob
        return 0


class ClusterBuilder:
    """
    Builds Cluster objects from a Population by dividing it into clusters
    and then using a specific ZoneBuilder to create zones within each cluster.
    """

    def __init__(self,
                 n_clusters: int,
                 zone_builder: BaseZoneBuilder,
                 split_size: float = 0.01):
        """
        Initializes the ClusterBuilder.

        Args:
            n_clusters (int): The number of clusters to form. Must be a positive integer.
            zone_builder (BaseZoneBuilder): An initialized instance of a ZoneBuilder
                                            (e.g., ClusteringZoneBuilder or SweepingZoneBuilder).
            split_size (float): A parameter (between 0 and 1) used by the AuxiliaryBalancedKMeans
                                 algorithm to determine the size of auxiliary splits during clustering.
                                 A smaller value leads to more balanced clusters.
        """
        self._n_clusters = n_clusters
        self._zone_builder = zone_builder
        self._split_size = split_size
        self._next_cluster_id = 0  # To assign unique IDs to clusters, initialized to 0

        if not isinstance(self._n_clusters, int) or self._n_clusters <= 0:
            raise ValueError("num_clusters must be a positive integer.")
        if not isinstance(self._zone_builder, BaseZoneBuilder):
            raise TypeError("zone_builder must be an instance of BaseZoneBuilder (or its subclass).")

    def _get_next_cluster_id(self) -> int:
        """
        Generates and returns a unique integer ID for a new cluster.
        This method ensures each cluster created by the builder has a distinct identifier.
        """
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        return cluster_id

    def build_clusters(self, population: Population) -> Tuple[List[Cluster], np.ndarray, np.ndarray]:
        """
        Builds a list of Cluster objects by first applying balanced k-means
        clustering to the entire population, and then using the configured
        ZoneBuilder to create zones within each resulting cluster.

        Args:
            population (Population): The overall Population object containing all
                                     units to be clustered and zoned.

        Returns:
            List[Cluster]: A list of Cluster objects, each containing
                           its assigned zones.
        """
        self._next_cluster_id = 0  # Reset the cluster ID counter for a new build operation

        # Initialize and fit the AuxiliaryBalancedKMeans model to the population
        # This step performs the initial division of the population into self._num_clusters
        upb_kmeans = UPBalancedKMeans(k=self._n_clusters, split_size=self._split_size)
        upb_kmeans.fit(population)

        clusters: List[Cluster] = []
        for cluster in upb_kmeans.clusters:
            free_indices = cluster['free']

            index_share = np.column_stack((free_indices, np.ones(free_indices.size)))

            # Use the provided ZoneBuilder instance to build zones for the units
            # belonging to the current cluster. The `index_share` array contains
            # the specific units and their shares for this cluster.
            zones_for_cluster = self._zone_builder.build_zones(population, index_share)

            # Generate a unique ID for the new cluster
            new_cluster_id = self._get_next_cluster_id()

            floor_index = cluster['border']['floor_index']
            floor_pct = cluster['border']['floor_percentage']
            ceil_index = cluster['border']['ceil_index']
            ceil_pct = cluster['border']['ceil_percentage']

            floor = np.array([floor_index, floor_pct], dtype=float) if floor_index != -1 else None
            ceil = np.array([ceil_index, ceil_pct], dtype=float) if ceil_index != -1 else None

            # Create a new Cluster object with its unique ID and the zones built for it
            new_cluster = Cluster(
                id=new_cluster_id,
                _pop=population,
                _zones=zones_for_cluster,
                _floor=floor,
                _ceil=ceil,
            )

            # Add the newly created cluster to the list of constructed clusters
            clusters.append(new_cluster)

        return clusters, upb_kmeans.labels, upb_kmeans.centroids


def generate_index_shares(n: int, membership: np.ndarray, source_indices: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """
    Prepares a list of `index_share` arrays, where each array corresponds to a cluster or zone
    and contains the original population indices and their corresponding membership shares.

    Args:
        n (int): The number of index_shares to generate.
        membership (np.ndarray): A 2D NumPy array representing the membership of each
                                 population unit to each cluster, as determined by
                                 AuxiliaryBalancedKMeans. The rows correspond to
                                 population unit indices, and columns correspond to cluster labels.
        source_indices (Optional[np.ndarray]): A 1D NumPy array representing the indices of source.

    Returns:
        List[np.ndarray]: A list of NumPy arrays. Each inner array is an `index_share`
                          array for a specific cluster. It has two columns: the first
                          column contains the original indices of the population units
                          belonging to that cluster, and the second column contains
                          their membership share (e.g., 1 for full membership in a
                          hard clustering, or a float for soft clustering).
    """
    # Get the row and column indices of non-zero elements in the membership matrix.
    # 'indices' will contain population unit indices, and 'labels' will contain cluster labels.
    indices, labels = np.nonzero(membership)
    index_shares = []

    output_indices = source_indices[indices] if source_indices is not None else indices

    # Iterate through each cluster to gather its members and their shares
    for i in range(n):
        # Create a boolean mask to select population units that belong to the current cluster (i)
        mask = (labels == i)

        # Get the population indices that are part of the current cluster
        masked_indices = indices[mask]
        masked_output_indices = output_indices[mask]

        # Get the membership shares for these specific population units in the current cluster
        # This uses advanced indexing to fetch the share from the original membership matrix
        shares = membership[(masked_indices, i)]

        # Combine the population indices and their membership shares into a 2-column array.
        # This `index_share` array is then appended to the list.
        index_shares.append(np.column_stack([masked_output_indices, shares]))
    return index_shares