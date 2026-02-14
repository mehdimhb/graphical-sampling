import numpy as np
from typing import Tuple, Union
from abc import ABC, abstractmethod


class Order(ABC):
    """
    Abstract Base Class for defining ordering strategies.

    Concrete subclasses must implement the `order` method.
    The `__call__` method is implemented to allow instances of
    Order subclasses to be called directly as functions,
    delegating to their `order` method.
    """

    @abstractmethod
    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Calculates the sorting order for a given set of 2D points.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices (shape (n,)) that,
                        when used to index the original `points` array,
                        would sort them according to the strategy.
        """
        pass

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Allows an instance of an Order subclass to be called directly
        as if it were a function. This delegates the call to the `order` method.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: The sorted indices as returned by the `order` method.
        """
        return self.order(points)


class Change(Order):
    def __init__(self, num_changes: int):
        self.num_changes = num_changes

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Changes the order of points randomly.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        indices = np.arange(points.shape[0])
        for num_change in range(self.num_changes):
            i, j = np.random.choice(indices.shape[0], 2)
            indices[i], indices[j] = indices[j], indices[i]

            # print(i, j)

        return np.argsort(indices)


class HilbertCurve(Order):
    """
    An ordering strategy that sorts points based on their position along
    a Hilbert space-filling curve.

    This strategy maps 2D coordinates to a 1D Hilbert curve index, then
    sorts the points based on these indices. It requires specifying the
    bounds of the coordinate space and the 'level' of the Hilbert curve.
    """

    def __init__(self,
                 min_coord: Union[np.ndarray, Tuple[float, float]],  # shape (2,)
                 max_coord: Union[np.ndarray, Tuple[float, float]],  # shape (2,)
                 level: int = 16):
        """
        Initializes the HilbertCurve strategy.

        Args:
            min_coord (Union[np.ndarray, Tuple[float, float]]): The minimum
                                                                (x, y) coordinates
                                                                of the bounding box.
            max_coord (Union[np.ndarray, Tuple[float, float]]): The maximum
                                                                (x, y) coordinates
                                                                of the bounding box.
            level (int): The level of the Hilbert curve. Higher levels provide
                         finer granularity and better approximation of spatial
                         proximity, but may increase computational cost.
                         Must be a positive integer.

        Raises:
            ValueError: If `level` is not a positive integer, or if `min_coord`
                        or `max_coord` are not 2-element arrays/tuples.
        """
        if not isinstance(level, int) or level <= 0:
            raise ValueError("Hilbert curve 'level' must be a positive integer.")

        min_coord = np.asarray(min_coord, dtype=np.float64)
        max_coord = np.asarray(max_coord, dtype=np.float64)

        if min_coord.shape != (2,) or max_coord.shape != (2,):
            raise ValueError("min_coord and max_coord must be 2-element arrays or tuples.")

        if np.any(min_coord >= max_coord):
            raise ValueError("min_coord must be strictly less than max_coord in both dimensions.")

        self._min_coord = min_coord
        self._max_coord = max_coord
        self._level = level

    def _coords_to_hilbert_indices(self, coords: np.ndarray) -> np.ndarray:
        """
        Converts 2D coordinates to 1D Hilbert curve indices using a Morton code
        approximation. This is a placeholder for a true Hilbert curve
        implementation which would typically involve more complex bitwise
        operations or a dedicated library.

        Args:
            coords (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of Morton curve indices.
        """
        # Normalize coordinates to [0, 1] range
        normalized_coords = (coords - self._min_coord) / (self._max_coord - self._min_coord)

        # Scale to the grid size (2^level x 2^level)
        grid_size = 2 ** self._level
        scaled_coords = np.floor(normalized_coords * grid_size).astype(int)

        # Clamp values to ensure they are within the valid grid range [0, grid_size - 1]
        scaled_coords = np.clip(scaled_coords, 0, grid_size - 1)

        # Morton code (Z-order curve) for 2D points
        morton_indices = np.empty(scaled_coords.shape[0], dtype=np.int64)
        for i, (x, y) in enumerate(scaled_coords):
            morton_code = 0
            for j in range(self._level):
                morton_code |= ((x >> j) & 1) << (2 * j)
                morton_code |= ((y >> j) & 1) << (2 * j + 1)
            morton_indices[i] = morton_code
        return morton_indices

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points based on their Hilbert curve (Morton code approximation) indices.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points
                        along the Hilbert curve.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        hilbert_indices = self._coords_to_hilbert_indices(points)
        # Return the indices that would sort the Hilbert indices
        return np.argsort(hilbert_indices)


class LexicoXY(Order):  # Renamed
    """
    An ordering strategy that sorts points lexicographically (first by x-coordinate,
    then by y-coordinate for ties).
    Corresponds to 'lexico-xy'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points lexicographically (x then y).

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points
                        lexicographically.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        # np.lexsort sorts by the last array first, then the second to last, etc.
        # So, to sort by x then y, we pass y then x.
        return np.lexsort((points[:, 1], points[:, 0]))


class LexicoYX(Order):  # Renamed
    """
    An ordering strategy that sorts points lexicographically (first by y-coordinate,
    then by x-coordinate for ties).
    Corresponds to 'lexico-yx'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points lexicographically (y then x).

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points
                        lexicographically.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        # To sort by y then x, we pass x then y.
        return np.lexsort((points[:, 0], points[:, 1]))


class Random(Order):  # Renamed
    """
    An ordering strategy that shuffles points randomly.
    Corresponds to 'random'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points randomly.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of randomly permuted integer indices.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")
        return np.random.permutation(points.shape[0])


class Angle(Order):  # Renamed
    """
    An ordering strategy that sorts points based on their angle relative
    to the positive x-axis, centered at the origin (0,0). Angles are
    normalized to be within [0, 2*pi).
    Corresponds to 'angle_0'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points by their angle around the origin.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points by angle.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        # arctan2 returns angles in (-pi, pi].
        # np.mod(..., 2 * np.pi) normalizes them to [0, 2*pi).
        angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
        return np.argsort(angles)


class DistFromOrigin(Order):  # Renamed
    """
    An ordering strategy that sorts points based on their Euclidean distance
    from the origin (0,0).
    Corresponds to 'distance_0'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points by their Euclidean distance from the origin.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points by distance.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        distances = np.linalg.norm(points, axis=1)
        return np.argsort(distances)


class Projection(Order):  # Renamed
    """
    An ordering strategy that sorts points based on the sum of their x and y coordinates.
    This can be interpreted as sorting by projection onto the line y = x.
    Corresponds to 'projection'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points by the sum of their coordinates (x + y).

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points by projection.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        projections = points[:, 0] + points[:, 1]
        return np.argsort(projections)


class DistFromCentroid(Order):  # Renamed
    """
    An ordering strategy that sorts points based on their Euclidean distance
    from the centroid (mean of all points) of the input set.
    Corresponds to 'center'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points by their Euclidean distance from the set's centroid.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points by distance from centroid.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")
        if points.shape[0] == 0:
            return np.array([], dtype=int)

        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        return np.argsort(distances)


class Spiral(Order):  # Renamed
    """
    An ordering strategy that sorts points in an approximate spiral pattern
    around their centroid.
    Corresponds to 'spiral'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points in a spiral pattern.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points in a spiral.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")
        if points.shape[0] <= 1:
            return np.arange(points.shape[0])

        centroid = points.mean(axis=0)

        # Translate points so centroid is at origin
        translated_points = points - centroid

        # Calculate angles relative to the centroid
        angles = np.mod(np.arctan2(translated_points[:, 1], translated_points[:, 0]), 2 * np.pi)

        # Calculate distances from the centroid
        distances = np.linalg.norm(translated_points, axis=1)

        # Sort primarily by angle, then by distance for ties in angle
        return np.lexsort((distances, angles))  # Moved inside the class


class MaxCoord(Order):  # Renamed
    """
    An ordering strategy that sorts points based on the maximum of their
    x or y coordinate (i.e., `max(x, y)`).
    Corresponds to 'max'.
    """

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points by the maximum of their x and y coordinates.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points by max coordinate.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")

        max_coords = np.max(points, axis=1)
        return np.argsort(max_coords)


class Snake(Order):  # Renamed
    """
    An ordering strategy that sorts points in a 'snake' or 'boustrophedon' pattern.
    It sorts primarily by the x-coordinate, and for every second "row" (defined by
    unique primary coordinate values), it reverses the order of the secondary coordinate.
    By default, it sorts by x then alternates y.
    """

    def __init__(self, axis_to_snake: int = 0):
        """
        Initializes the Snake strategy.

        Args:
            axis_to_snake (int): The axis along which the snake pattern should occur.
                                 0 for x-axis (sorts primarily by x, alternates y),
                                 1 for y-axis (sorts primarily by y, alternates x).
                                 Defaults to 0.
        Raises:
            ValueError: If `axis_to_snake` is not 0 or 1.
        """
        if axis_to_snake not in [0, 1]:
            raise ValueError("axis_to_snake must be 0 (x-axis) or 1 (y-axis).")
        self._axis_to_snake = axis_to_snake

    def order(self, points: np.ndarray) -> np.ndarray:
        """
        Orders points in a snake pattern.

        Args:
            points (np.ndarray): A 2D array of points (shape (n, 2)).

        Returns:
            np.ndarray: A 1D array of integer indices that sort the points
                        in a snake pattern.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a 2D array with shape (n, 2).")
        if points.shape[0] == 0:
            return np.array([], dtype=int)

        primary_coords = points[:, self._axis_to_snake]

        primary_sort_indices = np.argsort(primary_coords)

        final_indices = np.zeros_like(primary_sort_indices)
        current_idx = 0

        unique_primary_values, starts = np.unique(primary_coords[primary_sort_indices], return_index=True)

        for i, _ in enumerate(unique_primary_values):
            start_index = starts[i]
            end_index = starts[i + 1] if i + 1 < len(unique_primary_values) else len(points)

            segment_primary_sort_indices = primary_sort_indices[start_index:end_index]
            segment_secondary_coords = points[segment_primary_sort_indices, 1 - self._axis_to_snake]

            if i % 2 == 0:
                local_sort_within_segment = np.argsort(segment_secondary_coords)
            else:
                local_sort_within_segment = np.argsort(-segment_secondary_coords)

            final_indices[current_idx: current_idx + len(local_sort_within_segment)] = \
                segment_primary_sort_indices[local_sort_within_segment]

            current_idx += len(local_sort_within_segment)

        return final_indices
