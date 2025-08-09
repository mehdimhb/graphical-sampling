import numpy as np
from k_means_constrained import KMeansConstrained
from typing import Optional, List, Tuple
from ..population import Population


class AuxiliaryBalancedKMeans:
    """
    Implements an Auxiliary Balanced K-Means algorithm designed to cluster data points
    while considering associated probabilities and ensuring balanced cluster sizes.

    This algorithm first expands the input data points based on their probabilities,
    then applies a constrained K-Means algorithm to the expanded dataset. Finally,
    it maps the cluster assignments back to the original data points, determining
    their membership probabilities for each cluster.

    Args:
        k (int): The desired number of clusters.
        split_size (float, optional): A scaling factor used to expand the data points
                                      based on their probabilities. Smaller values lead
                                      to more expanded points. Defaults to 0.001.
    """

    def __init__(self, k: int, split_size: float = 0.001):
        self.k: int = k
        self.split_size: float = split_size
        self.N: Optional[int] = None  # Number of original data points
        self.membership: Optional[np.ndarray] = None  # Membership probabilities for each point to each cluster
        self.labels: Optional[np.ndarray] = None  # Final cluster labels for original points
        self.centroids: Optional[np.ndarray] = None  # Centroids for each cluster

    def _generate_expanded_coords(self, coords: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates expanded coordinates and their original indices based on probabilities.

        Each original data point is repeated a number of times proportional to its
        probability, scaled by `self.split_size`. This effectively gives more weight
        to points with higher probabilities during the constrained K-Means step.

        Args:
            coords (np.ndarray): A 2D NumPy array where each row is a coordinate of a data point.
            probs (np.ndarray): A 1D NumPy array of probabilities corresponding to each data point.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - expanded_coords (np.ndarray): The new array of expanded coordinates.
                - expanded_idx (np.ndarray): An array mapping each expanded coordinate back
                                            to its original index in `coords`.
                - original_point_expansion_counts (np.ndarray): The number of times each original
                                                                point was expanded.
        """
        # Calculate how many times each point should be repeated
        # Round to nearest integer and ensure at least one repeat
        original_point_expansion_counts: np.ndarray = (probs / self.split_size).round().astype(int)
        original_point_expansion_counts[original_point_expansion_counts == 0] = 1

        # Repeat the coordinates and their original indices based on calculated counts
        expanded_coords: np.ndarray = np.repeat(coords, original_point_expansion_counts, axis=0)
        expanded_idx: np.ndarray = np.repeat(np.arange(self.N), original_point_expansion_counts)

        return expanded_coords, expanded_idx, original_point_expansion_counts

    def _generate_membership(self, extended_labels: np.ndarray, expanded_idx: np.ndarray,
                             original_point_expansion_counts: np.ndarray) -> np.ndarray:
        """
        Calculates the membership probabilities for each original data point to each cluster.

        This is done by counting how many times each expanded representation of an original
        point was assigned to a particular cluster, and then normalizing these counts
        by the total number of expanded representations for that original point.

        Args:
            extended_labels (np.ndarray): Labels assigned to the expanded coordinates by KMeansConstrained.
            expanded_idx (np.ndarray): An array mapping each expanded coordinate back
                                       to its original index in the original dataset.
            original_point_expansion_counts (np.ndarray): The number of times each original
                                                          point was expanded.

        Returns:
            np.ndarray: A 2D NumPy array (N, k) where N is the number of original points
                        and k is the number of clusters. Each element (i, j) represents
                        the probability that original point 'i' belongs to cluster 'j'.
        """
        # Create a matrix to count how many times each expanded point (from original_idx)
        # was assigned to each cluster (extended_label)
        membership_counts_matrix: np.ndarray = np.zeros((self.N, self.k), dtype=int)
        np.add.at(membership_counts_matrix, (expanded_idx, extended_labels), 1)

        # Divide the counts by the total number of expanded points for each original point
        # This gives the proportion (membership probability)
        # Using np.newaxis to enable broadcasting for division
        membership: np.ndarray = membership_counts_matrix / original_point_expansion_counts[:, np.newaxis]
        return membership

    def fit(self, population: Population) -> None:
        """
        Fits the Auxiliary Balanced K-Means model to the given population data.

        Args:
            population (Population): An instance of the Population class containing
                                     the coordinates, probabilities, and optional IDs
                                     of the data points.
        """
        # Extract data from the Population object
        coords: np.ndarray = population.coords
        probs: np.ndarray = population.probs

        self.N = population.N

        # Generate expanded coordinates based on probabilities
        expanded_coords, expanded_idx, original_point_expansion_counts = self._generate_expanded_coords(coords, probs)

        # Apply KMeansConstrained to the expanded dataset
        # Calculate ideal cluster size for the constrained K-Means
        # Ensure that cluster_size is at least 1 to avoid issues with KMeansConstrained
        cluster_size: int = max(1, len(expanded_idx) // self.k)

        # Initialize and fit KMeansConstrained
        # n_jobs=-1 uses all available CPU cores for parallel processing
        kmeans = KMeansConstrained(
            n_clusters=self.k,
            size_min=cluster_size,
            size_max=cluster_size+1 if self.k > 1 else cluster_size,  # Allows for slight variation in cluster sizes
            n_jobs=-1,
            random_state=42 # For reproducibility
        )
        # extended_labels are the cluster assignments for the expanded points
        extended_labels: np.ndarray = kmeans.fit_predict(expanded_coords)

        # Calculate membership probabilities for each original point to each cluster
        self.membership = self._generate_membership(extended_labels, expanded_idx, original_point_expansion_counts)

        # Map cluster assignments back to original data points
        # Determine the final labels for the original data points
        self.labels: np.ndarray = np.argmax(self.membership, axis=1)

        # Compute centroids for each cluster based on original coordinates and labels
        self.centroids: np.ndarray = np.array([coords[self.labels == i].mean(axis=0) for i in range(self.k)])
