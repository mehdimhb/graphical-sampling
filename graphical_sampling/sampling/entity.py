from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Union, Optional

from ..population import Population
from .order import Order

# Type alias for arrays storing unit indices and their shares
IndexShareArray = np.ndarray  # expected shape (n, 2)
# First column: 0-based index of the unit within the Population's arrays.
# Second column: share/weight for that unit.


@dataclass(slots=True)
class Zone:
    """
    Represents a geographical or logical zone containing a set of sampling units
    and their associated shares within this specific zone.

    Attributes:
        id (int): A unique integer identifier for the zone.
        _pop (Population): The internal Population object this zone belongs to.
        _index_share (IndexShareArray): Internal NumPy array for unit indexes and their shares.
    """
    id: int
    _pop: Population # Internal storage for Population
    _index_share: IndexShareArray # Internal storage for index_share

    def __post_init__(self):
        # Validate that the initial _index_share has the correct shape
        if self._index_share.ndim != 2 or self._index_share.shape[1] != 2:
            raise ValueError("index_share must be a 2D array with shape (n, 2)")

    def __len__(self) -> int:
        """
        Returns the number of unique units within this zone (n from (n, 2)).
        """
        return self._index_share.shape[0]

    def __eq__(self, other: object) -> bool:
        """
        Compares two Zone objects for equality.

        Two zones are considered equal if their `index_share` arrays have the same values.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: True if the zones are equal, False otherwise.
        """
        if not isinstance(other, Zone):
            return NotImplemented
        return np.array_equal(self.index_share, other.index_share)

    # Public properties for controlled access
    @property
    def pop(self) -> Population:
        """
        Returns the Population object associated with this zone.
        This property is read-only.
        """
        return self._pop

    @property
    def index_share(self) -> IndexShareArray:
        """
        Returns a read-only view of the unit indices and their shares within this zone.
        Modifications to the returned array will raise an error.
        """
        view = self._index_share.view()
        view.flags.writeable = False
        return view

    # convenience views (no copies)
    @property
    def index(self) -> np.ndarray:  # (n,) int64
        """
        Returns a read-only view of the 0-based indices of the units
        within the global Population's arrays.

        These indices allow direct lookup of unit properties (like coordinates)
        from a `Population` object.

        Returns:
            np.ndarray: A 1D array of integer unit indices (into Population arrays).
        """
        return self._index_share[:, 0].astype(np.int64, copy=False)

    @property
    def share(self) -> np.ndarray:  # (n,) float64
        """
        Returns a read-only view of the shares (proportions/weights) of each
        unit within this zone.

        These shares represent the contribution of each unit to the zone's
        overall index (e.g., proportion of area, population, or index of size).

        Returns:
            np.ndarray: A 1D array of float shares.
        """
        return self._index_share[:, 1]

    @property
    def prob(self) -> np.ndarray:  # (n,) float64
        """
        Returns a read-only view of the probabilities of each
        unit within this zone.

        Returns:
            np.ndarray: A 1D array of float probs.
        """
        return self.share * self.pop.probs[self.index]

    @property
    def centroid(self) -> np.ndarray:
        """
        Calculates the arithmetic centroid (average coordinate) of the zone
        based on the coordinates of its constituent units, using the Population
        instance stored internally.

        This method does not use the 'share' attribute as a weighting factor
        for the centroid calculation, treating all constituent units as
        having equal weight for this geometric center.

        Returns:
            np.ndarray: A 1D NumPy array of shape (2,) representing the
                        (x, y) coordinates of the zone's arithmetic centroid.
        """
        # Retrieve coordinates of units within this zone using their population indices and internal _pop
        pts = self._pop.coords[self.index]
        # Calculate the mean along axis 0 (i.e., mean of x and mean of y)
        return np.mean(pts, axis=0)

    def apply_order(self, order: Order, inplace: bool = True) -> Zone | None:
        """
        Applies an ordering strategy to the `_index_share` of this zone

        Args:
            order (Order): A callable that takes a 2D NumPy array
                                            of coordinates (shape (n, 2)) and returns
                                            a 1D NumPy array of integer indices (shape (n,))
                                            that define the desired sort order.
            inplace (bool): If True, the ordering is applied in-place to the current Zone object
                            and the method returns None. If False, a new Zone object with the
                            applied ordering is returned, and the original Zone remains unchanged.

        Returns:
            Zone | None: Returns a new Zone object with the applied ordering if `inplace` is False,
                         otherwise returns None (as the operation is performed in-place).
        """
        # Get the coordinates of the units belonging to this zone using self._pop
        unit_coords = self._pop.coords[self.index]
        # Get the sorting indices from the strategy
        sort_indices = order(unit_coords)

        if inplace:
            # Apply the sorting in-place to _index_share
            self._index_share = self._index_share[sort_indices]
            return None
        else:
            # Create a new _index_share array with the applied sorting
            new_index_share = self._index_share[sort_indices].copy()
            # Return a new Zone object with the sorted data
            return Zone(id=self.id, _pop=self._pop, _index_share=new_index_share)

    def __hash__(self) -> int:
        data = self.index_share.tobytes()
        return hash((self.index_share.shape,
                     str(self.index_share.dtype),
                     data))


@dataclass(slots=True) # NOT frozen, to allow in-place modification by its own methods
class Cluster:
    """
    Represents a cluster of contiguous or related zones.

    Attributes:
        id (int): A unique integer identifier for the cluster.
        _zones (List[Zone]): Internal list of Zone objects that constitute this cluster.
    """
    id: int
    _pop: Population
    _zones: List[Zone] = field(default_factory=list) # Internal storage for zones
    _floor: Optional[IndexShareArray] = None
    _ceil: Optional[IndexShareArray] = None

    def __post_init__(self):
        """
        Performs validation after initialization to ensure consistency.
        Checks that all zones within the cluster refer to the same Population instance.
        """
        if self._zones: # Only perform check if there are zones
            for i, zone in enumerate(self._zones):
                # Ensure all zones refer to the exact same Population object (identity check)
                if zone.pop is not self._pop:
                    raise ValueError(
                        "Cluster and all zones within a cluster must refer to the same Population instance. "
                        f"Zone at index {i} has a different population reference."
                    )

    def __len__(self) -> int:
        """
        Returns the number of zones within this cluster.
        """
        return len(self._zones)

    def __eq__(self, other: object) -> bool:
        """
        Compares two Cluster objects for equality.

        Two clusters are considered equal if they contain the same zones in the same order.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: True if the clusters are equal, False otherwise.
        """
        if not isinstance(other, Cluster):
            return NotImplemented
        return self._zones == other._zones and np.array_equal(self._floor, other._floor) and np.array_equal(self._ceil, other._ceil)

    @property
    def pop(self) -> Population:
        """
        Returns the Population object associated with this cluster.
        This property is read-only.
        """
        return self._pop

    # Public property for controlled access
    @property
    def zones(self) -> List[Zone]:
        """
        Returns a shallow copy of the list of Zone objects in this cluster.
        Modifications to the returned list will not affect the internal list.
        To modify the internal list, use `apply_order_strategy`.
        """
        return list(self._zones) # Return a shallow copy to prevent external modification

    @property
    def _has_floor(self) -> bool:
        if self._floor is None:
            return False
        return self._floor.size > 0

    @property
    def _has_ceil(self) -> bool:
        if self._ceil is None:
            return False
        return self._ceil.size > 0

    @property
    def floor(self) -> Optional[IndexShareArray]:
        if not self._has_floor:
            return None
        view = self._floor.view()
        view.flags.writeable = False
        return view

    @property
    def ceil(self) -> Optional[IndexShareArray]:
        if not self._has_ceil:
            return None
        view = self._ceil.view()
        view.flags.writeable = False
        return view

    def get_index_share(self, reduce: bool = True) -> IndexShareArray:
        """
        Aggregates the unit population indices and shares from all constituent
        zones within this cluster.

        Args:
            reduce (bool, optional): If True (default), duplicate unit population
                                     indices across different zones within the
                                     cluster are merged, and their shares are summed.
                                     If False, a raw vertical concatenation
                                     of all unit index-share pairs from all zones
                                     is returned, potentially with duplicates.

        Returns:
            IndexShareArray: A NumPy array of shape (m, 2). The first
                             column contains 0-based unit indices
                             into the Population array, and the
                             second column contains their aggregated
                             shares within the cluster.
                             If `reduce` is True, each unit index
                             appears exactly once.
        """
        if not self._zones:
            combined = np.empty((0, 2), dtype=np.float64)
            if self._has_floor:
                combined = np.vstack([self._floor, combined])
            if self._has_ceil:
                combined = np.vstack([combined, self._ceil])
            return combined

        # Vertically stack all index_share from the zones in this cluster using _zones
        combined = np.vstack([zone.index_share for zone in self._zones])
        if self._has_floor:
            combined = np.vstack([self._floor, combined])
        if self._has_ceil:
            combined = np.vstack([combined, self._ceil])
        if not reduce:
            return combined

        # Reduce duplicate unit *indices* by summing their shares
        u, inv = np.unique(combined[:, 0].astype(np.int64), return_inverse=True)
        s = np.zeros_like(u, dtype=np.float64)
        np.add.at(s, inv, combined[:, 1])
        return np.column_stack((u, s))

    def get_total_prob_stats(self, zones_stats: bool = False) -> str:
        cluster_index_share = self.get_index_share(reduce=True)
        cluster_index = cluster_index_share[:, 0].astype(np.int64)
        cluster_share = cluster_index_share[:, 1]
        total_prob_for_cluster = float(np.sum(self._pop.probs[cluster_index] * cluster_share))
        stats = f"Total Prob in Cluster: {round(total_prob_for_cluster, 4)}"
        if zones_stats:
            for zone in self._zones:
                stats += f"\nTotal Prob in Zone {zone.id}: {round(np.sum(zone.prob), 4)}"
        return stats

    @property
    def zones_edges(self) -> np.ndarray:
        """
        Calculates the cumulative sum of probabilities for zones within the cluster.

        This property can be used to define "edges" or boundaries based on the
        probabilistic contribution of each zone.

        Returns:
            np.ndarray: A 1D NumPy array representing the cumulative probabilities
                        of zones. The first element is always 0, and the last
                        element is the total probability of all zones in the cluster.
        """
        edges = [0]
        start = 0
        for zone in self._zones:
            edges.append(start + np.sum(zone.prob))
            start += np.sum(zone.prob)
        return np.array(edges)

    @property
    def centroid(self) -> np.ndarray:
        """
        Calculates the arithmetic centroid (average coordinate) of the cluster.

        This calculation is based on the coordinates of all unique units
        aggregated from its constituent zones. It does not use the 'share'
        attribute as a weighting factor for the centroid calculation,
        treating all unique constituent units as having equal weight for
        this geometric center.

        Returns:
            np.ndarray: A 1D NumPy array of shape (2,) representing the
                        (x, y) coordinates of the cluster's arithmetic centroid.
        """
        if not self._zones:
            # If no zones, return NaNs
            return np.array([np.nan, np.nan], dtype=np.float64)

        # Get aggregated shares. 'reduce=True' ensures unique unit indices.
        aggregated_shares = self.get_index_share(reduce=True)

        # Handle case where aggregated shares could be empty (e.g., zones with no units)
        if aggregated_shares.size == 0:
            return np.array([np.nan, np.nan], dtype=np.float64)

        # Get the unique unit population indices from the first column
        unique_unit_pop_indices = aggregated_shares[:, 0].astype(np.int64)
        # Retrieve coordinates of these unique units using the Population object
        pts = self._pop.coords[unique_unit_pop_indices]
        # Calculate the mean along axis 0 (i.e., mean of x and mean of y)
        return np.mean(pts, axis=0)

    def apply_order(self, order: Order, inplace: bool = True) -> Cluster | None:
        """
        Applies an ordering strategy to the `_zones` list of this cluster.

        Args:
            order (Order): A callable that takes a 2D NumPy array
                                            of centroids (shape (n, 2)) and returns
                                            a 1D NumPy array of integer indices (shape (n,))
                                            that define the desired sort order.
            inplace (bool): If True, the ordering is applied in-place to the current Cluster object
                            and the method returns None. If False, a new Cluster object with the
                            applied ordering is returned, and the original Cluster remains unchanged.

        Returns:
            Cluster | None: Returns a new Cluster object with the applied ordering if `inplace` is False,
                            otherwise returns None (as the operation is performed in-place).
        """
        if not self._zones:
            return None if inplace else Cluster(id=self.id, _zones=[], _pop=self._pop, _floor=self._floor, _ceil=self._ceil)

        # Calculate centroids for all zones within this cluster.
        zone_centroids = np.array([zone.centroid for zone in self._zones])

        # Get the sorting indices from the strategy
        sort_indices = order(zone_centroids)

        if inplace:
            # Apply the sorting in-place to the list of zones using slice assignment
            self._zones[:] = [self._zones[i] for i in sort_indices]
            return None
        else:
            # Create a new list of zones with the applied sorting
            new_zones = [self._zones[i] for i in sort_indices]
            # Return a new Cluster object with the sorted zones
            return Cluster(id=self.id, _pop=self._pop, _zones=new_zones, _floor=self._floor, _ceil=self._ceil)

    def __hash__(self) -> int:
        floor = self._floor.tobytes() if self._has_floor else None
        ceil = self._ceil.tobytes() if self._has_ceil else None
        return hash((tuple(self._zones), floor, ceil))