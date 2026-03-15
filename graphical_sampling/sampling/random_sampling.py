import numpy as np
from numpy.typing import NDArray


class RandomSampling:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        n: int,
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.n = n
        self.rng = np.random.default_rng()

    def sample(self, n_samples: int):
        return np.array(
            [
                self.rng.integers(0, self.probs.shape[0], size=self.n)
                for _ in range(n_samples)
            ],
            dtype=int,
        )
