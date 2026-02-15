from ..clustering import FIPBalancedNMeans

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed


class Density:
    def __init__(
        self,
        population,
        k: int,
        labels: np.ndarray = None,
        centroids: np.ndarray = None,
        n_jobs: int = -1
    ):
        self.population = population
        self.coords = population.coords
        self.probs = population.probs
        self.original_density = self._density(self.coords)

        if (centroids is None) or (labels is None):
            self.labels, self.centroids = self._generate_labels_centroids(k)
        else:
            self.labels, self.centroids = labels, centroids

        self.n_jobs = n_jobs

    def _density(self, coords: np.ndarray) -> np.ndarray:
        kde = KernelDensity(kernel="tophat", bandwidth="scott").fit(coords)
        return np.exp(kde.score_samples(coords))

    def _generate_labels_centroids(self, k):
        fbn = FIPBalancedNMeans(n=k)
        fbn.fit(self.population)
        return fbn.labels, fbn.centroids

    def _assign_samples_to_centroids(self, samples, centroids):
        cost = np.linalg.norm(
            samples[:, :, np.newaxis] - centroids, axis=3
        ).transpose(0, 2, 1)

        return np.stack([
            samples[i][linear_sum_assignment(cost[i])[1]]
            for i in range(samples.shape[0])
        ], axis=0)

    def _generate_translated_coords(self, sample: np.ndarray) -> np.ndarray:
        translated = self.coords.copy()
        for j, translation in enumerate(sample - self.centroids):
            translated[self.labels == j] += translation
        return translated

    def _scale(self, arr, max_val):
        return np.clip(arr / max_val, -1, 1)

    def _score_density(self, translated_density: np.ndarray):
        norm = np.sqrt(2) * np.sqrt(self.original_density**2 + translated_density**2)
        spread = np.mean(
            self._scale(
                (self.original_density - translated_density)/norm,
                np.sin(np.pi/8)
            )
        )
        var = np.mean(
            self._scale(
                1 - (self.original_density + translated_density)/norm,
                1 - np.cos(np.pi/8)
            )
        )
        score = spread + (np.sign(spread) - spread) * var
        return score

    def _score_one_sample(self, sample: np.ndarray):
        translated = self._generate_translated_coords(sample)
        translated_density = self._density(translated)
        score = self._score_density(translated_density)
        return score, (self.original_density, translated_density)

    def score(self, samples: np.ndarray, return_densities: bool = False):
        """
        samples: array of shape (n_samples, total_points)
                 where each row is an index array into self.coords.
        """
        raw_samples = self.coords[samples]
        samples_assigned = self._assign_samples_to_centroids(raw_samples, self.centroids)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_one_sample)(s) for s in samples_assigned
        )

        scores, densities = zip(*results)
        scores = np.array(scores)

        if return_densities:
            return scores, list(densities)
        return scores
