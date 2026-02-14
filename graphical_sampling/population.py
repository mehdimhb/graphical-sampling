import numpy as np
from typing import Union, Optional


class Population:
    """
    Represents a population of sampling units, each with a unique identifier,
    2D spatial coordinates, and an associated first-order inclusion probability.

    This class is designed for use in sampling designs and provides efficient
    storage and access to population data using NumPy arrays. It ensures
    specific data types and shapes for its attributes to ensure data integrity
    and optimize performance in sampling operations.

    Attributes:
        _ids (np.ndarray): A 1D array of unique integer identifiers for each
                           sampling unit (shape (N,)).
        _coords (np.ndarray): A 2D array of float coordinates for each sampling unit (shape (N, 2)).
        _probs (np.ndarray): A 1D array of float values representing the
                             first-order inclusion probability for each sampling unit (shape (N,)).
        _indices (np.ndarray): A 1D array of original population indices (shape (N,)).
    """

    __slots__ = ("_ids", "_coords", "_probs", "_indices")

    def __init__(
        self,
        coords: np.ndarray,         # shape (N, 2)
        probs:  np.ndarray,         # shape (N,)
        ids:    Optional[np.ndarray] = None,  # shape (N,) or None, with a default value
        indices: Optional[np.ndarray] = None,
    ):
        """
        Initializes a new Population instance.

        Args:
            coords (np.ndarray): A 2D array of float spatial coordinates
                                 (shape (N, 2)).
            probs (np.ndarray): A 1D array of float first-order inclusion
                                probabilities (shape (N,)).
            ids (Optional[np.ndarray], optional): A 1D array of unique integer identifiers
                                               (shape (N,)). If None (default),
                                               `np.arange(N)` will be used to generate IDs.
            indices (Optional[np.ndarray], optional): A 1D array of original population indices (shape (N,)).
                                                      this is used in the case of subset populations.

        Raises:
            ValueError: If the shapes of `coords` or `probs` do not match the
                        implied population size N, or if `coords` is not (N,2).
        """
        N = coords.shape[0]

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be a 2D array with shape (N, 2)")
        if probs.ndim != 1 or probs.size != N:
            raise ValueError("probs must be a 1D array with shape (N,)")

        # Check that all values in probs are between 0 and 1
        if not np.all((probs >= 0) & (probs <= 1)):
            raise ValueError("All values in 'probs' must be between 0 and 1 (inclusive).")

        # Handle ids being None (which will be the case if it wasn't provided)
        if ids is None:
            _ids_final = np.arange(1, N+1, dtype=np.int64)
        else:
            if ids.ndim != 1 or ids.size != N:
                raise ValueError("ids must be a 1D array with shape (N,) or None")
            _ids_final = ids.astype(np.int64, copy=False)

        # Assign to slots, casting to canonical dtypes without copying when possible
        self._ids    = _ids_final
        # Normalize coordinates during initialization
        self._coords = np.ascontiguousarray(self._normalize_coords(coords), dtype=np.float64)
        self._probs  = np.ascontiguousarray(probs, dtype=np.float64)
        self._indices = np.ascontiguousarray(indices, dtype=np.in64) if indices is not None else None

    @staticmethod
    def _normalize_coords(coords: np.ndarray) -> np.ndarray:
        """
        Normalizes 2D coordinates to a range of [0, 1] based on their
        maximum overall span.

        The normalization is performed by:
        1. Subtracting the minimum value of each coordinate dimension.
        2. Dividing by the maximum span (range) across all dimensions.

        This ensures that the normalized coordinates fit within a unit square.

        Args:
            coords (np.ndarray): A 2D NumPy array of shape (N, 2) representing
                                 the spatial coordinates.

        Returns:
            np.ndarray: A 2D NumPy array of the same shape as `coords` with
                        normalized coordinates.
        """
        span = np.ptp(coords, axis=0)
        scale = span.max()
        if scale == 0:
            return np.zeros_like(coords, dtype=np.float64)
        return (coords - coords.min(axis=0)) / scale

    # ---------- read-only views enforced by properties ----------
    @property
    def ids(self) -> np.ndarray:
        """
        Returns a read-only view of the sampling unit identifiers.
        Returns:
            np.ndarray: A 1D array of integer IDs.
        """
        view = self._ids.view()
        view.flags.writeable = False
        return view

    @property
    def coords(self) -> np.ndarray:
        """
        Returns a read-only view of the sampling unit coordinates.
        Returns:
            np.ndarray: A 2D array of float coordinates.
        """
        view = self._coords.view()
        view.flags.writeable = False
        return view

    @property
    def probs(self) -> np.ndarray:
        """
        Returns a read-only view of the first-order inclusion probabilities.
        Returns:
            np.ndarray: A 1D array of float probabilities.
        """
        view = self._probs.view()
        view.flags.writeable = False
        return view

    @property
    def indices(self) -> np.ndarray:
        """
        Returns a read-only view of the indices.
        """
        if self._indices is None:
            return None
        view = self._indices.view()
        view.flags.writeable = False
        return view

    @property
    def x(self) -> np.ndarray:
        """
        Returns a read-only view of the x-coordinates of the sampling units.
        Returns:
            np.ndarray: A 1D array of float x-coordinates.
        """
        view = self._coords[:, 0].view()
        view.flags.writeable = False
        return view

    @property
    def y(self) -> np.ndarray:
        """
        Returns a read-only view of the y-coordinates of the sampling units.
        Returns:
            np.ndarray: A 1D array of float y-coordinates.
        """
        view = self._coords[:, 1].view()
        view.flags.writeable = False
        return view

    # ---------- basic utilities ----------
    @property
    def N(self) -> int:
        """
        Returns the total number of sampling units in the population.
        Returns:
            int: The population size (N).
        """
        return self._ids.size

    def sum_prob(self, idx: Optional[np.ndarray], share: Optional[np.ndarray]) -> float:
        if idx is None:
            return np.sum(self.probs)
        idx = idx.astype(np.int64)
        if share is None:
            return np.sum(self.probs[idx])
        return np.sum(self.probs[idx] * share)

    def as_stacked(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns a single (n,4) float64 block (id,p,x,y) for a specified
        subset of sampling units. This format is convenient for
        serialization or Pandas conversion, *without* altering the main copy.
        Args:
        idx (Optional[np.ndarray], optional): An array of indices (e.g., from a sample)
            to select a subset of the population. If None (default), the
            entire population will be considered.
        Returns:
        np.ndarray: A 2D array with 4 columns: ID, first-order inclusion
        probability, x-coordinate, and y-coordinate for the
        specified subset. The data type will be float64.
        """
        if idx is None:
            return np.column_stack((self._ids, self._probs, self._coords))
        idx = idx.astype(np.int64)
        return np.column_stack((self._ids[idx], self._probs[idx], self._coords[idx]))

    def subset(self, idx: np.ndarray, share: Union[float, np.ndarray] = 1.0) -> "Population":
        """
        Returns a new Population instance representing a view over a row subset.
        This operation is zero-copy when `idx` is a slice or a boolean mask,
        making it efficient for creating sub-populations.

        Args:
            idx (np.ndarray): An array of integer indices, a slice object,
                              or a boolean mask used to select a subset of
                              the population.
            share (Union[float, np.ndarray]): A scalar (float) or a NumPy array
                                             to be multiplied with the probabilities
                                             of the selected subset. If an array,
                                             it must be broadcastable with the
                                             selected probabilities. Defaults to 1.0.

        Returns:
            Population: A new Population instance containing the selected subset of individuals.

        Raises:
            ValueError: If `share` is a NumPy array and cannot be broadcast
                        with the probabilities of the selected subset.
        """
        # Select the subset of probabilities
        idx = idx.astype(np.int64)
        selected_probs = self._probs[idx]
        # Apply the share multiplier
        subset_probs = selected_probs * share
        # When creating a subset, the _ids should now be a concrete array, not None
        return Population(ids=self._ids[idx], coords=self._coords[idx], probs=subset_probs, indices=idx)
