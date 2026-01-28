import numpy as np
from numpy.typing import NDArray


#from . import Population
from ..population import Population

class KMeansSpatialSampling:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        n: int,
        n_zones: int | tuple[int, int],
        tolerance: int,
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.n = n
        self.n_zones = self._pair(n_zones)
        self.tolerance = tolerance

        self.population = Population(
            self.coords,
            self.probs,
            n_clusters=self.n,
            n_zones=self.n_zones,
            tolerance=self.tolerance,
        )
        self.rng = np.random.default_rng()

    def _pair(self, n: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(n, int):
            return (n, n)
        else:
            return n

    def sample(self, n_samples: int):
        samples = np.zeros((n_samples, self.n), dtype=int)
        for i in range(n_samples):
            random_number = self.rng.random()
            zone_index = np.searchsorted(
                np.arange(1, step=round(1 / np.prod(self.n_zones), self.tolerance)),
                random_number,
                side="right",
            )
            for j, cluster in enumerate(self.population.clusters):
                unit_index = np.searchsorted(
                    cluster.zones[zone_index - 1].units[:, 3].cumsum()
                    + (zone_index - 1)
                    * round(1 / np.prod(self.n_zones), self.tolerance),
                    random_number,
                    side="right",
                )
                if np.prod(self.n_zones) != len(cluster.zones):
                    warning_msg = (f"Warning: Cluster {j} has {len(cluster.zones)} zones, "
                                f"but expected {np.prod(self.n_zones)} based on n_zones={self.n_zones}")
                    print(warning_msg)
                samples[i, j] = cluster.zones[zone_index - 1].units[unit_index, 0]
        return samples
