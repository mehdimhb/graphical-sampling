from __future__ import annotations
import numpy as np


class Population:
    """
    Represents a pop of sampling units, each with a unique identifier,
    2D spatial coordinates, and an associated first-order inclusion prob.

    This class is designed for use in sampling designs and provides efficient
    storage and access to pop data using NumPy arrays. It ensures
    specific data types and shapes for its attributes to ensure data integrity
    and optimize performance in sampling operations.

    Attributes:
        _ids (np.ndarray): A 1D array of unique integer identifiers for each
                           sampling unit (shape (N,)).
        _coords (np.ndarray): A 2D array of float coordinates for each sampling unit (shape (N, 2)).
        _inclusions (np.ndarray): A 1D array of float values representing the
                             first-order inclusion prob for each sampling unit (shape (N,)).
        _indices (np.ndarray): A 1D array of original pop indices (shape (N,)).
    """

    __slots__ = ("_ids", "_coords", "_inclusions", "_variable", "_indices", "_n", "__weakref__")


    def __init__(
            self,
            coords: np.ndarray,  # shape (N, 2)
            inclusions: np.ndarray,  # shape (N,)
            variable: np.ndarray | None = None,  # shape (N,) or None
            ids: np.ndarray | None = None,  # shape (N,) or None
            indices: np.ndarray | None = None,  # shape (N,) or None
            n: int | None = None,  # target sample size for normalizing inclusions
    ):
        """
        Initializes a new Population instance.

        Args:
            coords (np.ndarray): A 2D array of float spatial coordinates
                                 (shape (N, 2)).
            inclusions (np.ndarray): A 1D array of float first-order inclusion
                                probabilities, or raw vector if `n` is provided (shape (N,)).
            variable (np.ndarray): A 1D array of float values representing the
                                    variable of interest for each sampling unit (shape (N,)) or None.
            ids (np.ndarray, optional): A 1D array of unique integer identifiers
                                               (shape (N,)). If None (default),
                                               `np.arange(1, N+1)` will be used to generate IDs.
            indices (np.ndarray, optional): A 1D array of original pop indices (shape (N,)).
                                                      this is used in the case of subset populations.
            n (int, optional): The desired sample size. If provided, `inclusions` is treated
                               as a vector of weights and normalized into probabilities.

        Raises:
            ValueError: If the shapes of `coords` or `inclusions` do not match the
                        implied pop size N, or if `coords` is not (N,2).
        """
        N = coords.shape[0]

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be a 2D array with shape (N, 2)")
        if inclusions.ndim != 1 or inclusions.size != N:
            raise ValueError("inclusions must be a 1D array with shape (N,)")

        # Normalize weights to inclusion probabilities if a sample size N is given
        if n is not None:
            inclusions = self._normalize_inclusions(inclusions, n)

        if not np.all(0 < inclusions) & np.all(inclusions <= 1):
            raise ValueError("All inclusion probabilities must be in the range (0, 1].")

        # Handle indices being None
        if ids is None:
            _ids_final = np.arange(1, N + 1, dtype=np.int64)
        else:
            if ids.ndim != 1 or ids.size != N:
                raise ValueError("indices must be a 1D array with shape (N,) or None")
            _ids_final = np.ascontiguousarray(ids, dtype=np.int64)

        if indices is None:
            _indices_final = np.arange(N, dtype=np.int64)
        else:
            if indices.ndim != 1 or indices.size != N:
                raise ValueError("indices must be a 1D array with shape (N,) or None")
            _indices_final = np.ascontiguousarray(indices, dtype=np.int64)

        self._ids = _ids_final
        self._coords = np.ascontiguousarray(self._normalize_coords(coords), dtype=np.float64)
        self._inclusions = np.ascontiguousarray(inclusions, dtype=np.float64)
        self._variable = np.ascontiguousarray(variable, dtype=np.float64)
        self._indices = _indices_final
        self._n = n if n is not None else int(np.sum(self._inclusions))

    @staticmethod
    def _normalize_inclusions(raw_inclusions: np.ndarray, n: int, max_iter: int = 1000) -> np.ndarray:
        """
        Calculates and adjusts inclusions so the sum of the probabilities equals `n`,
         while ensuring that no probability exceeds 1.
        """
        raw_inclusions = np.asarray(raw_inclusions, dtype=np.float64)
        if raw_inclusions.size == 0:
            raise ValueError("Input weights array is empty.")

        n_neg = np.sum(raw_inclusions < 0)
        n_null = np.sum(raw_inclusions == 0)

        if n_null > 0:
            raise ValueError("There are zero values in the initial weights.")
        if n_neg > 0:
            raise ValueError(f"There are {n_neg} negative value(s) shifted to zero.")

        total_weight = np.sum(raw_inclusions)
        if total_weight == 0:
            return np.zeros_like(raw_inclusions)

        live_count = int(np.count_nonzero(raw_inclusions > 0))
        n = max(0, min(n, live_count))

        if n == 0:
            return np.zeros_like(raw_inclusions)

        # Initial probabilities
        inclusions = n * raw_inclusions / total_weight

        # Iterative adjustment
        prev_maxed = -1
        for _ in range(max_iter):
            is_at_max = inclusions >= 1.0
            maxed_count = np.count_nonzero(is_at_max)

            if maxed_count in (0, prev_maxed):
                break
            prev_maxed = maxed_count

            mask = ~is_at_max
            total_x = np.sum(inclusions[mask])

            if total_x > 0:
                # In-place scaling of non-maxed elements
                inclusions[mask] *= (n - maxed_count) / total_x
            else:
                inclusions[mask] = 0.0

            inclusions[is_at_max] = 1.0
        # print('samiiii', inclusions.sum())
        return inclusions

    @staticmethod
    def _normalize_coords(coords: np.ndarray) -> np.ndarray:
        """
        Normalizes 2D coordinates to a range of [0, 1] based on their
        maximum overall span.
        """
        span = np.ptp(coords, axis=0)
        scale = span.max()
        if scale == 0:
            return np.zeros_like(coords, dtype=np.float64)
        return (coords - coords.min(axis=0)) / scale

    # ---------- read-only views enforced by properties ----------
    @property
    def ids(self) -> np.ndarray:
        view = self._ids.view()
        view.flags.writeable = False
        return view

    @property
    def coords(self) -> np.ndarray:
        view = self._coords.view()
        view.flags.writeable = False
        return view

    @property
    def inclusions(self) -> np.ndarray:
        view = self._inclusions.view()
        view.flags.writeable = False
        return view

    @property
    def variable(self) -> np.ndarray | None:
        if self._variable is None:
            return None
        view = self._variable.view()
        view.flags.writeable = False
        return view

    @property
    def indices(self) -> np.ndarray | None:
        if self._indices is None:
            return None
        view = self._indices.view()
        view.flags.writeable = False
        return view

    @property
    def x(self) -> np.ndarray:
        view = self._coords[:, 0].view()
        view.flags.writeable = False
        return view

    @property
    def y(self) -> np.ndarray:
        view = self._coords[:, 1].view()
        view.flags.writeable = False
        return view

    @property
    def n(self) -> int:
        return self._n

    @property
    def N(self) -> int:
        return self._ids.size

    # ---------- basic utilities ----------
    def sum_prob(self, idx: np.ndarray | None, share: np.ndarray | None) -> float:
        if idx is None:
            return float(np.sum(self.inclusions))
        idx = idx.astype(np.int64)
        if share is None:
            return float(np.sum(self.inclusions[idx]))
        return float(np.sum(self.inclusions[idx] * share))

    def as_stacked(self, idx: np.ndarray | None = None) -> np.ndarray:
        if idx is None:
            return np.column_stack((self._ids, self._inclusions, self._coords))
        idx = idx.astype(np.int64)
        return np.column_stack((self._ids[idx], self._inclusions[idx], self._coords[idx]))

    def subset(self, idx: np.ndarray, share: float | np.ndarray = 1.0) -> Population:
        idx = idx.astype(np.int64)

        subset_pop = Population.__new__(Population)
        subset_pop._coords = self._coords[idx]
        subset_pop._inclusions = self._inclusions[idx] * share
        subset_pop._ids = self._ids[idx]
        subset_pop._indices = idx
        subset_pop._n = self._n

        return subset_pop
