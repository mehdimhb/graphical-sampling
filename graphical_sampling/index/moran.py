import numpy as np
from scipy.spatial.distance import cdist

from ..population import Population


class Moran:
    def __init__(self, population: Population):
        self.population = population
        self.coords = self.population.coords
        self.inclusion_probs = self.population.inclusions

    @staticmethod
    def _calculate_spatial_weights(coords: np.ndarray, inclusion_probs: np.ndarray,
                                   bound: float = 1.0) -> np.ndarray:
        """
        Generates the stratification weight matrix W0.
        Optimized by pre-computing and pre-sorting all spatial distances simultaneously.
        """
        population_size = len(inclusion_probs)
        spatial_weights = np.zeros((population_size, population_size))
        epsilon = 1e-7

        # Pre-compute all distances and sort them at the C-level (Massive $O(N^2 \log N)$ speedup)
        dist_matrix = cdist(coords, coords, metric='euclidean')
        sorted_indices_matrix = np.argsort(dist_matrix, axis=1, kind='stable')

        for current_unit in range(population_size):
            distances = dist_matrix[current_unit]
            sorted_indices = sorted_indices_matrix[current_unit]

            cumulative_prob = 0.0
            j = 0

            # Find the inclusion probability cutoff bound
            while True:
                cumulative_prob += inclusion_probs[sorted_indices[j]]
                j += 1
                if (bound - cumulative_prob) <= epsilon or j >= population_size:
                    break

            last_added_idx = j - 1
            cutoff_distance = distances[sorted_indices[last_added_idx]]

            # Find all units that lie exactly on the cutoff distance
            tied_distance_indices = np.where(distances == cutoff_distance)[0]

            # Calculate the lower probability bound
            lower_bound_prob = cumulative_prob
            s = last_added_idx
            if sorted_indices[0] not in tied_distance_indices:
                while s >= 0 and sorted_indices[s] in tied_distance_indices:
                    lower_bound_prob -= inclusion_probs[sorted_indices[s]]
                    s -= 1

            # Calculate the upper probability bound
            upper_bound_prob = lower_bound_prob + np.sum(inclusion_probs[tied_distance_indices])

            # Apply proportional weights to units tied at the boundary
            unit_weights = np.zeros(population_size)
            if upper_bound_prob - lower_bound_prob > epsilon:
                proportion = (bound - lower_bound_prob) / (upper_bound_prob - lower_bound_prob)
                unit_weights[tied_distance_indices] = inclusion_probs[tied_distance_indices] * proportion

            # Keep exact inclusion probabilities for all units safely inside the bound
            for tt in range(s + 1):
                unit_weights[sorted_indices[tt]] = inclusion_probs[sorted_indices[tt]]

            spatial_weights[current_unit, :] = unit_weights

        return spatial_weights

    @staticmethod
    def _calculate_batch_moran(spatial_weights: np.ndarray, sample_indicators: np.ndarray) -> np.ndarray:
        """
        Calculates Moran's I for ALL samples simultaneously using pure matrix operations.
        sample_indicators: 2D array of shape (Population_Size, Number_of_Samples)
        """
        # Shape reference: N = population size, S = number of samples
        row_weight_sums = np.sum(spatial_weights, axis=1, keepdims=True)  # Shape: (N, 1)
        total_weight = np.sum(spatial_weights)  # Scalar

        # 1. Weighted Mean Sample -> Shape: (1, S)
        weighted_sample_means = np.sum(row_weight_sums * sample_indicators, axis=0, keepdims=True) / total_weight

        # 2. Centered Sample Data -> Shape: (N, S)
        centered_samples = sample_indicators - weighted_sample_means

        # 3. Core Matrix Multiplication (W @ z) -> Shape: (N, S)
        # This single operation replaces the need to loop through samples
        weighted_centered_sums = spatial_weights @ centered_samples

        # Numerator calculation -> Shape: (S,)
        numerator = np.sum(centered_samples * weighted_centered_sums, axis=0)

        # Denominator 1 (Variance of the samples) -> Shape: (S,)
        variance_samples = np.sum(row_weight_sums * (centered_samples ** 2), axis=0)

        # Denominator 2 (Variance of the spatial weights) -> Shape: (S,)
        with np.errstate(divide='ignore', invalid='ignore'):
            term1_matrix = (weighted_centered_sums ** 2) / row_weight_sums
            term1_matrix[row_weight_sums[:, 0] == 0, :] = 0  # Safe handling of zero-weight rows
            term1 = np.sum(term1_matrix, axis=0)

        term2 = (np.sum(weighted_centered_sums, axis=0) ** 2) / total_weight
        variance_weights = term1 - term2

        # Final calculation, preventing division by zero -> Shape: (S,)
        denominator = np.sqrt(variance_samples * variance_weights)

        scores = np.divide(numerator, denominator, out=np.full_like(numerator, np.inf), where=(denominator != 0))
        return scores

    def score(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculates Moran's I Index for a given array of samples.
        samples: 2D array where each row represents the selected unit indices for a single sample.
        """
        population_size = len(self.inclusion_probs)
        num_samples = len(samples)
        sample_size = samples.shape[1] if samples.ndim > 1 else len(samples)

        # 1. Precompute and format the spatial weight matrix W
        weight_matrix_w0 = self._calculate_spatial_weights(self.coords, self.inclusion_probs)
        spatial_weights = weight_matrix_w0.copy()
        np.fill_diagonal(spatial_weights, 0)

        # 2. Construct a 2D Indicator Matrix (One-Hot Encoding for all samples at once)
        # This replaces the Python `for` loop used to create masks.
        sample_indicators = np.zeros((population_size, num_samples))

        row_indices = samples.astype(int).flatten()
        col_indices = np.repeat(np.arange(num_samples), sample_size)
        sample_indicators[row_indices, col_indices] = 1

        # 3. Process all samples simultaneously
        moran_scores = self._calculate_batch_moran(spatial_weights, sample_indicators)

        return moran_scores
