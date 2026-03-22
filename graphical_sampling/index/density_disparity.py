from ..clustering import FIPBalancedNMeans

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed


class DensityDisparity:
    def __init__(
            self,
            population,
            n_jobs: int = -1,
            clustering_tol: float = 1e-9,
            clustering_max_iter: int = 100,
            kde_rtol: float = 1e-4
    ):
        self.pop = population
        self.coords = self.pop.coords
        self.probs = self.pop.inclusions

        self.n_jobs = n_jobs
        self.clustering_tol = clustering_tol
        self.clustering_max_iter = clustering_max_iter
        self.kde_rtol = kde_rtol

        # 1. First KDE: Calculate the original density once, as self.coords never changes
        self.original_density = self._density(self.coords)

        # --- Precompute and cache constants for scoring ---
        self._orig_density_sq = self.original_density ** 2
        self._sqrt_2 = np.sqrt(2)
        self._sin_pi_8 = np.sin(np.pi / 8)
        self._one_minus_cos_pi_8 = 1.0 - np.cos(np.pi / 8)

    def _density(self, coords: np.ndarray) -> np.ndarray:
        kde = KernelDensity(
            kernel="tophat",
            bandwidth="scott",
            rtol=self.kde_rtol
        ).fit(coords)
        return np.exp(kde.score_samples(coords))

    @staticmethod
    def _scale(arr, max_val):
        return np.clip(arr / max_val, -1.0, 1.0)

    def _score_single_density(self, translated_density: np.ndarray) -> float:
        # Adapted to process a 1D array instead of a 2D batch array
        norm = self._sqrt_2 * np.sqrt(self._orig_density_sq + translated_density ** 2)
        inv_norm = 1.0 / norm

        spread_arr = (self.original_density - translated_density) * inv_norm
        spread = np.mean(self._scale(spread_arr, self._sin_pi_8)).item()

        var_arr = 1.0 - (self.original_density + translated_density) * inv_norm
        var = np.mean(self._scale(var_arr, self._one_minus_cos_pi_8)).item()

        return spread + (np.sign(spread) - spread) * var

    def _process_one_sample(self, sample_indices: np.ndarray):
        """The complete, independent pipeline for a single sample."""
        raw_sample = self.coords[sample_indices]

        # 1. Clustering per sample (Using the raw_sample as initial centroids)
        fbn = FIPBalancedNMeans(
            n=self.pop.n,
            tol=self.clustering_tol,
            max_iter=self.clustering_max_iter,
            init_clust_method = 'expanded'
        )
        fbn.fit(self.pop, init_centroids=raw_sample)
        labels = fbn.labels
        centroids = fbn.centroids

        # 2. Assign sample points to the newly generated centroids
        cost = cdist(centroids, raw_sample)
        _, col_ind = linear_sum_assignment(cost)
        assigned_sample = raw_sample[col_ind]

        # 3. Translation
        translations = assigned_sample - centroids
        translated_coords = self.coords + translations[labels]

        # 4. Second KDE: DensityDisparity of the translated coordinates
        translated_density = self._density(translated_coords)

        # 5. Score Calculation
        score = self._score_single_density(translated_density)

        return score, translated_density

    def score(self, samples: np.ndarray, return_densities: bool = False):
        """
        samples: array of shape (n_samples, total_points)
                 where each row is an index array into self.coords.
        """
        # Distribute the full pipeline for each sample across available CPU cores
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_one_sample)(s) for s in samples
        )

        # Unpack the results
        scores, densities = zip(*results)
        scores = np.array(scores)

        if return_densities:
            return scores, [(self.original_density, td) for td in densities]
        return scores
