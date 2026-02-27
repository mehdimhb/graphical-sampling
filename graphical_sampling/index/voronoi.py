import numpy as np
from numba import njit, prange
from ..population import Population


@njit(parallel=True, fastmath=True)
def _compute_batch_voronoi_numba(coords: np.ndarray, probs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    JIT-compiled, multithreaded C-level scoring for Spatial Balance (Voronoi).
    Processes massive sample batches simultaneously across all CPU cores.
    """
    num_samples = samples.shape[0]
    sample_size = samples.shape[1]
    population_size = coords.shape[0]
    dimensions = coords.shape[1]

    scores = np.zeros(num_samples)

    # Process each sample independently in parallel
    for s_idx in prange(num_samples):
        sample_indices = samples[s_idx]

        # Array to hold the accumulated inclusion probabilities for this specific sample
        accumulated_incl = np.zeros(sample_size)

        # For every unit in the population, find the nearest sample unit(s)
        for i in range(population_size):
            dists_sq = np.zeros(sample_size)
            min_dist_sq = np.inf

            # 1. Calculate squared distances to all sample units
            for j in range(sample_size):
                sample_unit_idx = int(sample_indices[j])
                d_sq = 0.0
                for dim in range(dimensions):
                    d_sq += (coords[i, dim] - coords[sample_unit_idx, dim]) ** 2

                dists_sq[j] = d_sq
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq

            # 2. Count exact ties (using a tiny epsilon for float safety)
            ties = 0
            for j in range(sample_size):
                if dists_sq[j] <= min_dist_sq + 1e-11:
                    ties += 1

            # 3. Distribute the population unit's probability to the nearest sample unit(s)
            prob_share = probs[i] / ties
            for j in range(sample_size):
                if dists_sq[j] <= min_dist_sq + 1e-11:
                    accumulated_incl[j] += prob_share

        # 4. Calculate the final Variance/MSE around 1.0
        result = 0.0
        for j in range(sample_size):
            result += (accumulated_incl[j] - 1.0) ** 2

        scores[s_idx] = result / sample_size

    return scores


class Voronoi:
    def __init__(self, population: Population):
        self.population = population
        self.coords = self.population.coords
        self.probs = self.population.inclusions

    def score(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculates the Voronoi Spatial Balance Score for an array of samples.
        samples: 1D array (single sample) or 2D array (batch of samples).
        """
        # Ensure samples is a 2D array for batch processing
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # Ensure indices are integers
        samples = samples.astype(np.int64)

        # Process all samples in parallel
        voronoi_scores = _compute_batch_voronoi_numba(self.coords, self.probs, samples)

        return voronoi_scores
