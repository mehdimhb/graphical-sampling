import numpy as np
from numba import njit, prange
from ..population import Population


@njit(parallel=True, fastmath=True)
def _compute_batch_local_balance_numba(coords: np.ndarray, probs: np.ndarray, samples: np.ndarray,
                                       qq_inv: np.ndarray) -> np.ndarray:
    """
    JIT-compiled, multithreaded C-level scoring for Local Balance.
    Processes sample batches simultaneously, utilizing the precomputed qq_inv matrix.
    """
    num_samples = samples.shape[0]
    sample_size = samples.shape[1]
    population_size = coords.shape[0]
    dimensions = coords.shape[1]
    p1 = dimensions + 1

    scores = np.zeros(num_samples)

    # Process each sample independently in parallel
    for s_idx in prange(num_samples):
        sample_indices = samples[s_idx]

        # Track which units are currently in the sample
        in_sample = np.zeros(population_size, dtype=np.bool_)
        for j in range(sample_size):
            in_sample[int(sample_indices[j])] = True

        # xd stores the difference vectors between the sample units and the Voronoi set
        xd = np.zeros((sample_size, p1))

        # 1. Initialize xd with the sample units based on inclusion probabilities
        for j in range(sample_size):
            pop_idx = int(sample_indices[j])
            prob_factor = (1.0 - probs[pop_idx]) / probs[pop_idx]

            xd[j, 0] = prob_factor
            for dim in range(dimensions):
                xd[j, dim + 1] = coords[pop_idx, dim] * prob_factor

        # 2. Iterate through all population units to build the Voronoi domains
        for i in range(population_size):
            if in_sample[i]:
                continue

            # Find the nearest sample neighbor(s)
            dists_sq = np.zeros(sample_size)
            min_dist_sq = np.inf

            for j in range(sample_size):
                sample_unit_idx = int(sample_indices[j])
                d_sq = 0.0
                for dim in range(dimensions):
                    d_sq += (coords[i, dim] - coords[sample_unit_idx, dim]) ** 2

                dists_sq[j] = d_sq
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq

            # Count equidistant ties
            ties = 0
            for j in range(sample_size):
                if dists_sq[j] <= min_dist_sq + 1e-11:
                    ties += 1

            # Distribute the unit's value across the nearest sample neighbor(s)
            unit_val_0 = 1.0 / ties
            for j in range(sample_size):
                if dists_sq[j] <= min_dist_sq + 1e-11:
                    xd[j, 0] -= unit_val_0
                    for dim in range(dimensions):
                        xd[j, dim + 1] -= (coords[i, dim] / ties)

        # 3. Calculate the final Mahalanobis distance using the inverted qq matrix
        result = 0.0
        for j in range(sample_size):
            vec = xd[j, :]
            # Numba efficiently compiles np.dot into C-level matrix multiplications
            result += np.dot(vec, np.dot(qq_inv, vec))

        scores[s_idx] = np.sqrt(result / population_size)

    return scores


class LocalBalance:
    def __init__(self, population: Population):
        self.population = population
        self.coords = self.population.coords
        self.probs = self.population.inclusions

        # Precompute the `qq` matrix inverse strictly once for the entire population.
        # This replaces the C++ inefficiency of doing this per sample.
        N = len(self.coords)
        X_aug = np.hstack([np.ones((N, 1)), self.coords])
        qq = X_aug.T @ X_aug

        # Use pseudo-inverse (pinv) to guarantee stability even if coordinates are perfectly collinear
        self.qq_inv = np.linalg.pinv(qq)

    def score(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculates the Local Balance Score for an array of samples.
        samples: 1D array (single sample) or 2D array (batch of samples).
        """
        # Ensure samples is a 2D array for batch processing
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        samples = samples.astype(np.int64)

        # Process all samples in parallel
        local_balance_scores = _compute_batch_local_balance_numba(
            self.coords,
            self.probs,
            samples,
            self.qq_inv
        )

        return local_balance_scores
