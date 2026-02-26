import numpy as np
from numpy.typing import NDArray


class Generator:
    def __init__(self, seed: int = None) -> None:
        self.rng = np.random.default_rng(seed)

    def grid_coords(self, size: int | tuple[int, ...] = None) -> NDArray:
        """
        Generate grid coordinates.

        Args:
            size (int|Tuple[int, ...]]): Specifies the grid dimensions.
                - If None, generates a single grid of size `(1,)`.
                - If an integer N, generates a single grid of size `(N,)`.
                - If a tuple of `(N, D)`, generates a single grid with `N` points per `D` dimensions.
                  The size of the grids is `(N**D, D)`.
                - If a tuple of `(B, N, D)`, generates a batch of `B` grids of size `(N**D, D)`.

        Returns:
            NDArray: An array containing the coordinates of the grid(s), normalized to [0, 1].
        """
        batch, N, dim = self._check_size(size)

        linspace = np.linspace(0, 1, N)
        grid = np.meshgrid(*[linspace] * dim)
        base_coordinates = np.stack([indices.ravel() for indices in grid], axis=-1)
        coordinates = np.repeat(base_coordinates[np.newaxis, :, :], batch, axis=0)

        return coordinates.squeeze()

    def rand_coord(self, size: int | tuple[int, ...] = None) -> NDArray:
        """
        Generate random coordinates uniformly from [0, 1].

        Args:
            size (int|Tuple[int, ...]]): Specifies the grid dimensions.
                - If None, generates a single coordinate of size `(1,)`.
                - If an integer N, generates one dimensional coordinates for N points.
                - If a tuple of `(N, D)`, generates a D-dimensional coordinates for N points.
                - If a tuple of `(B, N, D)`, generates a batch of `B` coordinates of size `(N, D)`.

        Returns:
            NDArray: An array containing random coordinates in [0, 1].
        """
        batch, N, dim = self._check_size(size)
        coordinates = self.rng.random((batch, N, dim))
        return coordinates.squeeze()

    def uniform_coords(
        self, low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] = None
    ) -> NDArray:
        """
        Generate random coordinates uniformly from [low, high].

        Args:
            low (float): Lower boundary of the output interval. All values generated will be greater than or equal to low.
                The default value is 0.
            high (float): Upper boundary of the output interval. All values generated will be less than high.
                high - low must be non-negative. The default value is 1.0.
            size (int|Tuple[int, ...]]): Specifies the grid dimensions.
                - If None, generates a single coordinate of size `(1,)`.
                - If an integer N, generates one dimensional coordinates for N points.
                - If a tuple of `(N, D)`, generates a D-dimensional coordinates for N points.
                - If a tuple of `(B, N, D)`, generates a batch of `B` coordinates of size `(N, D)`.

        Returns:
            NDArray: An array containing random coordinates in [0, 1].
        """
        batch, N, dim = self._check_size(size)
        coordinates = self.rng.uniform(low, high, (batch, N, dim)).squeeze()
        return coordinates.squeeze()

    def normal_1d_coords(
        self, mean: float = 0.0, std: float = 1.0, size: int | tuple[int, ...] = None
    ) -> NDArray:
        """
        Generate random 1D coordinates sampled from a normal distribution.

        Args:
            mean (float): Mean of the normal distribution (default 0).
            std (float): Standard deviation of the normal distribution (default 1).
            size (int|Tuple[int, ...]]): Specifies the coordinates dimensions.
                - If None, generates a single coordinate of size `(1,)`.
                - If an integer N, generates one dimensional coordinates for N points.
                - If a tuple of `(B, N)`, generates a batch of `B` coordinates of size `(N,)`.
                - If a tuple of `(B, N, D)`, D has to be 1.

        Returns:
            NDArray: An array containing coordinates sampled from a normal distribution.
        """
        batch, N, dim = self._check_size(size, fixed_dim=1)
        coordinates = self.rng.normal(mean, std, (batch, N)).squeeze()
        return coordinates.squeeze()

    def normal_md_coords(
        self, mean: NDArray, cov: NDArray, size: int | tuple[int, ...] = None
    ) -> NDArray:
        """
        Generate random m-dimensional coordinates sampled from a multivariate normal distribution.

        Args:
            mean (ArrayLike): Mean of the m-dimensional distribution.
            cov (ArrayLike): Covariance matrix of the distribution.
            size (int|Tuple[int, ...]]): Specifies the coordinates dimensions.
                - If None, generates a single coordinate of size `(1, m)`.
                - If an integer N, generates m-dimensional coordinates for N points. the shape of coordinates is `(N, m)`.
                - If a tuple of `(B, N)`, generates a batch of `B` coordinates of size `(N, m)`.
                - If a tuple of `(B, N, D)`, D has to be m.

        Returns:
            NDArray: An array containing coordinates sampled from a multivariate normal distribution.
        """
        batch, N, dim = self._check_size(size, fixed_dim=mean.shape[0])
        coordinates = self.rng.multivariate_normal(mean, cov, (batch, N)).squeeze()
        return coordinates.squeeze()

    def cluster_coords(
        self,
        n_clusters: int,
        cluster_std: float | NDArray,
        size: int | tuple[int, ...] = None,
    ) -> NDArray:
        """
        Generate clustered coordinates by placing points around cluster centers.

        Args:
            n_clusters (int): Number of clusters to generate.
            cluster_std (float|NDArray): Standard deviation of points around each cluster center.
            size (int|Tuple[int, ...]]): Specifies the grid dimensions.
                - If None, generates a single coordinate of size `(1,)`.
                - If an integer N, generates one dimensional coordinates for N points.
                - If a tuple of `(N, D)`, generates a D-dimensional coordinates for N points.
                - If a tuple of `(B, N, D)`, generates a batch of `B` coordinates of size `(N, D)`.

        Returns:
            NDArray: An array containing clustered coordinates.
        """
        if not isinstance(n_clusters, int):
            raise ValueError("n_clusters must be an integer.")
        if not isinstance(cluster_std, (float, np.ndarray)):
            raise ValueError("cluster_std must be a float or an array.")

        batch, N, dim = self._check_size(size)

        cluster_centers = self.rng.random((batch, n_clusters, dim))
        cluster_assignments = self.rng.integers(0, n_clusters, size=(batch, N))
        coordinates = np.zeros((batch, N, dim))

        for b in range(batch):
            for n in range(N):
                cluster_id = cluster_assignments[b, n]
                coordinates[b, n, :] = self.rng.normal(
                    cluster_centers[b, cluster_id],
                    cluster_std
                    if isinstance(cluster_std, float)
                    else cluster_std[cluster_id],
                    dim,
                )

        return coordinates.squeeze()

    def equal_inclusions(
        self, n: int, size: int | tuple[int, ...] = None
    ) -> NDArray:
        """
        Generate equal probabilities that sum up to n.

        Args:
            n (int): The total sum of the probabilities.
            size (int): The size of probabilities.
                - If None, generates a single prob of size `(1,)`.
                - If an integer N, generates N probabilities.
                - If a tuple of `(B, N)`, generates a batch of `B` probabilities of size `(N,)`.
                - If a tuple of `(B, N, D)`, D has to be 1.

        Returns:
            NDArray: An array of size `size` with equal probabilities summing to `n`.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be an positive integer.")

        batch, N, dim = self._check_size(size, fixed_dim=1)
        probabilities = np.full((batch, N, dim), n / N)
        return probabilities.squeeze()

    def unequal_inclusions(
        self, n: int, size: int | tuple[int, ...] = None
    ) -> NDArray:
        """
        Generate equal probabilities that sum up to n.

        Args:
            n (int): The total sum of the probabilities.
            size (int): The size of probabilities.
                - If None, generates a single prob of size `(1,)`.
                - If an integer N, generates N probabilities.
                - If a tuple of `(B, N)`, generates a batch of `B` probabilities of size `(N,)`.
                - If a tuple of `(B, N, D)`, D has to be 1.

        Returns:
            NDArray: An array of size `size` with equal probabilities summing to `n`.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be an positive integer.")

        batch, N, dim = self._check_size(size, fixed_dim=1)
        probabilities = self.rng.random((batch, N))
        probabilities *= n / probabilities.sum(axis=1)
        return probabilities.squeeze()

    @staticmethod
    def _check_size(size: int | tuple[int, ...], fixed_dim: int = None):
        """
        Validate and parse the size input.

        Args:
            size (int|Tuple[int, ...]]): Specifies the dimensions.

        Returns:
            Tuple[int, int, int]: A tuple `(B, N, D)` where:
                - `B` is the batch size.
                - `N` is the number of points per in the batch (or per dimension in some cases).
                - `D` is the dimensionality of points.
        """
        if isinstance(size, int):
            return 1, size, fixed_dim or 1
        if size is None or len(size) == 0:
            return 1, 1, fixed_dim or 1
        if not isinstance(size, tuple) or len(size) > 3:
            raise ValueError("Size must be a tuple of at most 3 positive integers.")
        if not all(isinstance(x, int) and x > 0 for x in size):
            raise ValueError("All elements of size must be positive integers.")
        if len(size) == 3 and fixed_dim is not None and size[2] != fixed_dim:
            raise ValueError(
                "The dimension is fixed. you cannot set a different dimension."
            )

        match len(size):
            case 1:
                return 1, *size, fixed_dim or 1
            case 2:
                return (1, *size) if fixed_dim is None else (*size, fixed_dim)
            case 3:
                return size
        return None


def rng(seed=None):
    return Generator(seed=seed)
