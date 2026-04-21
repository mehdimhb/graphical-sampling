import numpy as np
from scipy.spatial.distance import cdist

from ..population import Population


class Moran:
    def __init__(self, population: Population, method: str = "tille"):
        self.population = population
        self.coords = self.population.coords
        self.inclusion_probs = self.population.inclusions
        self.method = method.lower()

    @staticmethod
    def _weights_tille(coords, inclusion_probs):
        N = len(inclusion_probs)
        W = np.zeros((N, N))
        eps = 1e-12

        D = cdist(coords, coords)

        for i in range(N):
            pi_i = inclusion_probs[i]

            if pi_i >= 1 - eps:
                continue

            h = min(1.0 / pi_i - 1.0, N - 1)
            k = int(np.floor(h))
            frac = h - k

            order = np.argsort(D[i])
            order = order[order != i]

            if k > 0:
                W[i, order[:k]] = 1.0

            if k < (N - 1):
                W[i, order[k]] = frac

            s = W[i].sum()
            if s > eps:
                W[i] /= s

        return W

    @staticmethod
    def _weights_robertson(coords, inclusion_probs):
        N = len(inclusion_probs)
        W = np.zeros((N, N))
        eps = 1e-12
        D_mat = cdist(coords, coords) # renaming to avoid confusion with matrix D

        for i in range(N):
            pi_i = inclusion_probs[i]
            if pi_i >= 1 - eps:
                continue

            # CORRECT FORMULA: h = min(1/pi - 1, N - 1) 
            h = min(1.0 / pi_i - 1.0, N - 1)
            k = int(np.floor(h))
            frac = h - k

            order = np.argsort(D_mat[i])
            order = order[order != i] # Exclude self [cite: 83, 87]

            if k > 0:
                W[i, order[:k]] = 1.0

            if k < (N - 1):
                W[i, order[k]] = frac

            # REMOVED: Row Normalization Block
            # s = W[i].sum()
            # if s > eps:
            #     W[i] /= s

        return W

    @staticmethod
    def _weights_raphael(coords, inclusion_probs, bound=1.0):
        N = len(inclusion_probs)
        W = np.zeros((N, N))
        eps = 1e-7

        D = cdist(coords, coords)
        sorted_idx = np.argsort(D, axis=1)

        for i in range(N):

            cumulative = 0.0
            j = 0

            while True:
                cumulative += inclusion_probs[sorted_idx[i, j]]
                j += 1
                if (bound - cumulative) <= eps or j >= N:
                    break

            last = j - 1
            cutoff_dist = D[i, sorted_idx[i, last]]

            tied = np.where(D[i] == cutoff_dist)[0]

            lower = cumulative
            s = last

            if sorted_idx[i, 0] not in tied:
                while s >= 0 and sorted_idx[i, s] in tied:
                    lower -= inclusion_probs[sorted_idx[i, s]]
                    s -= 1

            upper = lower + np.sum(inclusion_probs[tied])

            weights = np.zeros(N)

            if upper - lower > eps:
                prop = (bound - lower) / (upper - lower)
                weights[tied] = inclusion_probs[tied] * prop

            for t in range(s + 1):
                weights[sorted_idx[i, t]] = inclusion_probs[sorted_idx[i, t]]

            W[i] = weights

        return W

    @staticmethod
    def _calculate_batch_moran(W: np.ndarray, indicators: np.ndarray) -> np.ndarray:
        row_sums = np.sum(W, axis=1, keepdims=True)
        total_w = np.sum(W)

        weighted_means = np.sum(row_sums * indicators, axis=0, keepdims=True) / total_w
        Z = indicators - weighted_means
        WZ = W @ Z

        numerator = np.sum(Z * WZ, axis=0)
        var1 = np.sum(row_sums * (Z ** 2), axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = (WZ ** 2) / row_sums
            tmp[row_sums[:, 0] == 0, :] = 0
            t1 = np.sum(tmp, axis=0)

        t2 = (np.sum(WZ, axis=0) ** 2) / total_w
        var2 = t1 - t2

        denom = np.sqrt(var1 * var2)

        scores = np.divide(
            numerator,
            denom,
            out=np.full_like(numerator, np.inf),
            where=(denom != 0)
        )

        return scores

    def score(self, samples: np.ndarray) -> np.ndarray:

        N = len(self.inclusion_probs)
        S = len(samples)
        sample_size = samples.shape[1]

        if self.method == "tille":
            W = self._weights_tille(self.coords, self.inclusion_probs)

        elif self.method == "raphael":
            W = self._weights_raphael(self.coords, self.inclusion_probs)

        elif self.method == "robertson":
            W = self._weights_robertson(self.coords, self.inclusion_probs)

        else:
            raise ValueError("method must be 'tille', 'raphael', or 'robertson'")

        np.fill_diagonal(W, 0)

        indicators = np.zeros((N, S))

        r = samples.astype(int).flatten()
        c = np.repeat(np.arange(S), sample_size)

        indicators[r, c] = 1

        return self._calculate_batch_moran(W, indicators)
