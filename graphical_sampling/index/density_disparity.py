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
            kde_rtol: float = 1e-4,
            representative: str = "nmeans",   # "nmeans" or "nmedoids"
    ):
        self.pop = population
        self.coords = self.pop.coords
        self.probs = self.pop.inclusions

        self.n_jobs = n_jobs
        self.clustering_tol = clustering_tol
        self.clustering_max_iter = clustering_max_iter
        self.kde_rtol = kde_rtol

        if representative not in {"nmeans", "nmedoids"}:
            raise ValueError("representative must be either 'nmeans' or 'nmedoids'.")
        self.representative = representative

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

        # 1. Clustering per sample
        fbn = FIPBalancedNMeans(
            n=self.pop.n,
            tol=self.clustering_tol,
            max_iter=self.clustering_max_iter,
            init_clust_method="weighted"
        )
        fbn.fit(self.pop, init_centroids=raw_sample)

        labels = fbn.labels
        centroids = fbn.centroids

        # 2. Choose cluster representative: n-means or n-medoids
        if self.representative == "nmeans":
            reference_points = centroids

        elif self.representative == "nmedoids":
            reference_points, medoid_indices = self._cluster_medoids(labels, centroids)

        else:
            raise ValueError("representative must be either 'nmeans' or 'nmedoids'.")

        # 3. Assign sample points to the selected reference points
        cost = cdist(reference_points, raw_sample)
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_sample = np.empty_like(reference_points)
        assigned_sample[row_ind] = raw_sample[col_ind]

        # 4. Translation based on either centroids or medoids
        translations = assigned_sample - reference_points
        translated_coords = self.coords + translations[labels]

        # 5. Second KDE
        translated_density = self._density(translated_coords)

        # 6. Score calculation
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
    
    def _cluster_medoids(self, labels: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return one medoid for each cluster.

        The medoid is the population unit inside the cluster that minimizes
        the weighted sum of distances to all other units in that cluster.
        """
        K = centroids.shape[0]
        medoids = np.empty_like(centroids)
        medoid_indices = np.empty(K, dtype=int)

        for k in range(K):
            idx = np.flatnonzero(labels == k)

            if idx.size == 0:
                # Very rare fallback: keep the centroid if a cluster is empty
                medoids[k] = centroids[k]
                medoid_indices[k] = -1
                continue

            if idx.size == 1:
                medoid_indices[k] = idx[0]
                medoids[k] = self.coords[idx[0]]
                continue

            X = self.coords[idx]
            w = self.probs[idx]

            # Weighted medoid objective
            D = cdist(X, X, metric="euclidean")
            objective = D @ w

            min_obj = objective.min()
            candidate_locals = np.flatnonzero(np.isclose(objective, min_obj))

            # Tie-breaking 1: choose the candidate with the largest IP
            candidate_probs = self.probs[idx[candidate_locals]]
            max_prob = candidate_probs.max()
            candidate_locals = candidate_locals[np.isclose(candidate_probs, max_prob)]

            # Tie-breaking 2: if still tied, choose the smallest population index
            best_local = candidate_locals[np.argmin(idx[candidate_locals])]

            medoid_indices[k] = idx[best_local]
            medoids[k] = self.coords[medoid_indices[k]]
        return medoids, medoid_indices
