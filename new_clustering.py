import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


# Add class _Clustering to fip_balanced_nmeans.py
# and add _get_labels_centroids_ot_kmeans to class of FIPBalancedNMeans
# and edit this part of method 'fit' in the FIPBalancedNMeans class:
raw_labels, raw_centroids = self._get_labels_centroids_ot_kmeans(coords, probs, init_centroids)


def _get_labels_centroids_ot_kmeans(
        self,
        coords: np.ndarray,
        inclusions: np.ndarray,
        init_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    clustering = _Clustering(
        n_clusters=self.n,
        epsilon=0.01,
        max_iter=100,
        sinkhorn_iter=100,
        tol=1e-4,
        n_init=5,
    )
    clustering.fit(coords, inclusions, init_centroids)
    return clustering.labels, clustering.centroids


class _Clustering:
    def __init__(
        self,
        n_clusters: int,
        epsilon: float = 0.05,
        max_iter: int = 50,
        sinkhorn_iter: int =50,
        tol: float = 1e-4,
        n_init: int = 5,
        random_state: int | None = None,
    ):
        self.K = n_clusters
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sinkhorn_iter = sinkhorn_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.membership: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.objective: float | None = None

    @staticmethod
    def _compute_distances(coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return cdist(coords, centroids, metric='sqeuclidean')

    def _weighted_kmeans_pp(
            self, coords: np.ndarray, inclusions: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        N, D = coords.shape
        K = self.K

        centers = np.empty((K, D))

        idx = rng.integers(N)
        centers[0] = coords[idx]

        closest_dist = np.sum((coords - centers[0]) ** 2, axis=1)

        for k in range(1, K):
            probs = inclusions * closest_dist
            probs /= probs.sum()

            idx = rng.choice(N, p=probs)
            centers[k] = coords[idx]

            new_dist = np.sum((coords - centers[k]) ** 2, axis=1)
            closest_dist = np.minimum(closest_dist, new_dist)

        return centers

    def _sinkhorn_log(self, C, pi, b, f, g):

        eps = self.epsilon

        for _ in range(self.sinkhorn_iter):

            tmp = (g[None, :] - C) / eps
            f = eps * (np.log(pi) - logsumexp(tmp, axis=1))

            tmp = (f[:, None] - C) / eps
            g = eps * (np.log(b) - logsumexp(tmp, axis=0))

        return f, g

    @staticmethod
    def _hard_objective(coords: np.ndarray, centroids: np.ndarray, labels: np.ndarray, inclusion: np.ndarray) -> float:
        d = np.sum((coords - centroids[labels]) ** 2, axis=1)
        return np.sum(inclusion * d).item()

    def _fit_single(self, coords: np.ndarray, inclusion: np.ndarray, rng: np.random.Generator,
                    init_centroids: np.ndarray | None = None):
        N, D = coords.shape
        K = self.K
        b = np.ones(K)

        if init_centroids is not None:
            assert init_centroids.shape == (K, D), "init_centroids shape must be (n_clusters, n_features)"
            centroids = init_centroids.copy()
        else:
            centroids = self._weighted_kmeans_pp(coords, inclusion, rng)

        f = np.zeros(N)
        g = np.zeros(K)

        for _ in range(self.max_iter):
            C = self._compute_distances(coords, centroids)
            C = C / (np.median(C) + 1e-12)

            f, g = self._sinkhorn_log(C, inclusion, b, f, g)
            P = np.exp((f[:, None] + g[None, :] - C) / self.epsilon)

            new_centroids = P.T @ coords
            shift = np.linalg.norm(new_centroids - centroids)

            centroids = new_centroids
            if shift < self.tol:
                break

        R = P / inclusion[:, None]
        labels = np.argmax(R, axis=1)
        obj = self._hard_objective(coords, centroids, labels, inclusion)

        return centroids, R, labels, obj

    def fit(self, coords: np.ndarray, inclusions: np.ndarray,
            init_centroids: np.ndarray | None = None) -> _Clustering:
        rng = np.random.default_rng(self.random_state)

        best_obj = np.inf
        best_state = None

        n_runs = 1 if init_centroids is not None else self.n_init
        for _ in range(n_runs):
            centroids, R, labels, obj = self._fit_single(coords, inclusions, rng, init_centroids=init_centroids)
            if obj < best_obj:
                best_obj = obj
                best_state = (centroids, R, labels)

        self.centroids, self.membership, self.labels = best_state
        self.objective = best_obj
        return self