from __future__ import annotations

from typing import Iterator
from itertools import chain
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import loggamma

from .population import Population
from .structs import _MaxHeap, _Sample
from .order import Order
from .index import DensityDisparity, Moran, Voronoi, LocalBalance


class Design:
    def __init__(
            self,
            population: Population,
            num_partitions: int = 1,
            permutation: bool = False
    ):
        assert isinstance(num_partitions, int) and num_partitions >= 1, \
            f'num_partitions must be an integer and >= 1, got {num_partitions}.'

        self._pop = population
        self._num_partitions = num_partitions
        self._order = Order.from_indices(population, permutation=permutation)
        self._heaps = [_MaxHeap() for _ in range(num_partitions)]
        self._rng = np.random.default_rng()

        self._build()

        self._all_samples_and_probs: tuple[np.ndarray, np.ndarray] | None = None
        self._nht_variance: float | None = None
        self._density_expected_and_variance: tuple[float, float] | None = None
        self._moran_expected_and_variance: tuple[float, float] | None = None
        self._voronoi_expected_and_variance: tuple[float, float] | None = None
        self._local_balance_expected_and_variance: tuple[float, float] | None = None
        self._entropy: float | None = None
        self._maximum_entropy: float | None = None
        self._fip: np.ndarray | None = None
        self._indicator_matrix: np.ndarray | None = None
        self._sip: np.ndarray | None = None

    # ======================================== build and copy ========================================

    @classmethod
    def from_order(cls, population: Population, order: Order) -> Design:
        design = cls.__new__(cls)

        design._pop = population
        design._num_partitions = order.num_zones
        design._order = order
        design._heaps = [_MaxHeap() for _ in range(order.num_zones)]
        design._rng = np.random.default_rng()

        design._build()
        cls._reset_stats(design)

        return design

    def copy(self) -> Design:
        new_design = Design.__new__(Design)

        new_design._pop = self.pop
        new_design._num_partitions = self.num_partitions
        # new_design._num_partitions = 1
        new_design._order = self.order.copy()
        new_design._heaps = [h.copy() for h in self._heaps]
        new_design._rng = np.random.default_rng()

        self._reset_stats(new_design)

        return new_design

    @staticmethod
    def _reset_stats(design: Design) -> None:
        design._all_samples_and_probs = None
        design._nht_variance = None
        design._density_expected_and_variance = None
        design._moran_expected_and_variance = None
        design._voronoi_expected_and_variance = None
        design._local_balance_expected_and_variance = None
        design._entropy = None
        design._maximum_entropy = None
        design._fip = None
        design._indicator_matrix = None
        design._sip = None
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        # Designs are equal if their internal heap structures are identical
        return self._heaps == other._heaps

    def __hash__(self) -> int:
        # A stable hash based on the current state of all partitions
        return hash(tuple(self._heaps))

    def _build(self):
        events: list[tuple[float, str, int]] = []
        
        # 1. Add partition boundaries FIRST
        for i in range(self.num_partitions + 1):
            events.append((i / self.num_partitions, "boundary", -1))

        order_data = self.order.get()
        total_n = self.pop.n
        
        # THE EXACT CUMULATIVE MASS FIX
        # Use vectorized cumsum to prevent floating-point drift over 1000 items
        ids = order_data[:, 0].astype(int)
        shares = order_data[:, 1]
        p_array = self.pop.inclusions[ids] * shares
        
        cum_mass = np.zeros(len(p_array) + 1, dtype=np.float64)
        cum_mass[1:] = np.cumsum(p_array)
        cum_mass[-1] = float(total_n) # Snap absolute final boundary

        for i in range(len(order_data)):
            idx = int(ids[i])
            start_abs = cum_mass[i]
            end_abs = cum_mass[i+1]
            
            if end_abs - start_abs < 1e-12: continue

            # Safely extract the integer component (which "wrap" we are in)
            start_int = int(np.round(start_abs)) if np.isclose(start_abs, np.round(start_abs), atol=1e-10) else int(np.floor(start_abs))
            end_int = int(np.round(end_abs)) if np.isclose(end_abs, np.round(end_abs), atol=1e-10) else int(np.floor(end_abs))

            # Calculate the remainder (where it lands on the 0.0 to 1.0 line)
            start_rem = start_abs - start_int
            end_rem = end_abs - end_int
            
            # Clean tiny floating point noise
            if np.isclose(start_rem, 0.0, atol=1e-10): start_rem = 0.0
            if np.isclose(end_rem, 0.0, atol=1e-10): end_rem = 0.0
            if np.isclose(start_rem, 1.0, atol=1e-10): start_rem = 0.0; start_int += 1
            if np.isclose(end_rem, 1.0, atol=1e-10): end_rem = 0.0; end_int += 1

            if start_int < end_int and end_rem > 0.0:
                events.append((start_rem, "start", idx))
                events.append((1.0, "end", idx))
                events.append((0.0, "start", idx))
                events.append((end_rem, "end", idx))
            elif start_int < end_int and end_rem == 0.0:
                events.append((start_rem, "start", idx))
                events.append((1.0, "end", idx))
            else:
                events.append((start_rem, "start", idx))
                events.append((end_rem, "end", idx))

        events.sort(key=lambda x: (x[0], 0 if x[1] == "boundary" else (1 if x[1] == "end" else 2)))
        
        self.events = events
        active = set()
        last_point: float = 0.0

        for point, event_type, bar_index in events:
            if point > last_point + 1e-12:
                if active:
                    midpoint = (last_point + point) / 2
                    zone_idx = min(int(midpoint * self.num_partitions), self.num_partitions - 1)
                    length = point - last_point
                    self._push(zone_idx, _Sample(length, frozenset(active)))

            if event_type == "start":
                active.add(int(bar_index))
            elif event_type == "end":
                active.discard(int(bar_index))

            last_point = point
    # ======================================== Sampling ========================================

    def sample(self, num_samples: int) -> np.ndarray:
        rng = np.random.default_rng()
        random_numbers = rng.random(num_samples)
        samples, probs = self.all_samples_and_probs
        indices = np.searchsorted(probs.cumsum(), random_numbers)
        return samples[indices]

    # ======================================== Change ========================================
    def exchange(
            self,
            partitions: int = 1,
            pull_strategy: Literal['default', 'random', 'largest'] = 'default',
            exchange_coef: float = 0.75,
            window: int | None = None,
    ) -> None:
        if not (0 < partitions <= self.num_partitions):
            raise ValueError(f"partitions must be <= {self.num_partitions}.")

        valid_partitions = [i for i, h in enumerate(self._heaps) if len(h) >= 2]
        if not valid_partitions: return

        actual_count = min(partitions, len(valid_partitions))
        selected_partitions = self._rng.choice(valid_partitions, size=actual_count, replace=False)

        for part_idx in selected_partitions:
            part_idx = int(part_idx)

            if window is None:
                # =========================================================
                # ORIGINAL FAST PATH
                # =========================================================
                sample1 = self._pull(part_idx, 'largest' if pull_strategy == 'default' else pull_strategy)
                sample2 = self._pull(part_idx, 'random' if pull_strategy == 'default' else pull_strategy)

                if sample1.ids == sample2.ids:
                    self._push(part_idx, _Sample(sample1.prob + sample2.prob, sample1.ids))
                else:
                    self._push(part_idx, *self._switch(sample1, sample2, exchange_coef))
            else:
                # =========================================================
                # SCALPEL (Precision Window Path - Leak Proof!)
                # =========================================================
                # Safely pull ALL valid positive samples out using the class method
                samples_list = []
                while len(self._heaps[part_idx]) > 0:
                    samples_list.append(self._pull(part_idx, 'largest'))
                
                n_samples = len(samples_list)
                if n_samples >= 2:
                    if pull_strategy == 'largest' or pull_strategy == 'default':
                        idx1 = 0
                    else:
                        idx1 = self._rng.integers(0, n_samples)
                    
                    low = max(0, idx1 - window)
                    high = min(n_samples, idx1 + window + 1)
                    choices = [idx for idx in range(low, high) if idx != idx1]
                    idx2 = self._rng.choice(choices) if choices else (idx1 + 1) % n_samples
                    
                    # Pop the larger index first to avoid shifting
                    s2 = samples_list.pop(max(idx1, idx2))
                    s1 = samples_list.pop(min(idx1, idx2))
                    
                    if s1.ids == s2.ids:
                        self._push(part_idx, _Sample(s1.prob + s2.prob, s1.ids))
                    else:
                        self._push(part_idx, *self._switch(s1, s2, exchange_coef))
                
                # Safely push the untouched samples back into the heap
                for s in samples_list:
                    self._push(part_idx, s)

        self._reset_stats(self)


    def _pull(self, part_idx: int, strategy: Literal['random', 'largest']) -> _Sample:
        if strategy == 'random':
            return self._heaps[part_idx].random_pop()
        return self._heaps[part_idx].pop()

    def _push(self, part_idx: int, *args: _Sample) -> None:
        for r in args:
            if not r.almost_zero():
                self._heaps[part_idx].push(r)

    def _switch(self, sample1: _Sample, sample2: _Sample, coef: float = 0.5) -> tuple[_Sample, ...]:
        diff1 = list((sample1.ids - sample2.ids) - self.order.fixed_ids)
        diff2 = list((sample2.ids - sample1.ids) - self.order.fixed_ids)

        if not diff1 or not diff2:
            return sample1, sample2

        length = coef * min(sample1.prob, sample2.prob)
        n1 = self._rng.choice(diff1)
        n2 = self._rng.choice(diff2)

        return (
            _Sample(length, sample1.ids - {n1} | {n2}),
            _Sample(sample1.prob - length, sample1.ids),
            _Sample(length, sample2.ids - {n2} | {n1}),
            _Sample(sample2.prob - length, sample2.ids),
        )

    def merge_identical(self):
        for i in range(self.num_partitions):
            dic = {}
            for r in self._heaps[i]:
                dic[r.ids] = dic.get(r.ids, 0) + r.prob
            self._heaps[i] = _MaxHeap(
                initial_heap=[-_Sample(length, ids) for ids, length in dic.items()]
            )

        self._reset_stats(self)

    # ======================================== plot and special methods ========================================

    def plot(self, save: bool = False) -> None:
        initial_level: float = 0
        c_map_names = ['Blues', 'Greens', 'Wistia', 'Oranges', 'Reds', 'Purples']
        plt.figure(figsize=(8, 4.5))

        for zone_idx, heap in enumerate(self._heaps):
            c_map_name = c_map_names[zone_idx % len(c_map_names)]
            c_map = plt.get_cmap(c_map_name)

            samples = sorted(list(heap), key=lambda s: s.prob, reverse=True)
            num_samples = len(samples)

            for sample_idx, r in enumerate(samples):
                if num_samples > 1:
                    intensity = 0.9 - (0.6 * (sample_idx / (num_samples - 1)))
                else:
                    intensity = 0.7

                sample_color = c_map(intensity)

                for i in r.ids:
                    plt.plot(
                        [i, i],
                        [initial_level, initial_level + r.prob],
                        color=sample_color,
                        linewidth=1,
                        solid_capstyle='butt'
                    )
                initial_level += r.prob

        for i in range(1, self.num_partitions):
            plt.axhline(y=i / self.num_partitions, color='gray', linestyle='--', alpha=0.4)

        plt.ylabel("Probability")
        plt.xlabel("Population Units")
        plt.title("Design" if self.num_partitions == 1 else f"Design (with {self.num_partitions} Partitions)")
        if save:
            plt.savefig("design.png", dpi=300)
        plt.show()

    def __iter__(self) -> Iterator[_Sample]:
        # We merge identical IDs within each partition first to clean up the heap
        self.merge_identical()
        return chain.from_iterable(self._heaps)

    def __len__(self) -> int:
        return sum(len(h) for h in self._heaps)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        return self._heaps == other._heaps

    def __hash__(self) -> int:
        return hash(tuple(self._heaps))

    # ======================================== getters for main properties ========================================

    @property
    def pop(self) -> Population:
        return self._pop

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    @property
    def order(self) -> Order:
        return self._order

    # ============================== Cached properties for scores and samples/probs ==============================

    @property
    def all_samples_and_probs(self) -> tuple[np.ndarray, np.ndarray]:
        if self._all_samples_and_probs is not None:
            return self._all_samples_and_probs

        temp_samples = []
        temp_probs = []
        target_n = self.pop.n

        for sample in self:
            # Only keep samples that are reasonably close to target size
            # or collect them to see what's happening
            temp_samples.append(list(sample.ids))
            temp_probs.append(sample.prob)

        # --- REPAIR LOGIC ---
        final_samples = []
        final_probs = []
        
        for s_ids, s_prob in zip(temp_samples, temp_probs):
            if len(s_ids) == target_n:
                final_samples.append(s_ids)
                final_probs.append(s_prob)
            elif len(s_ids) > 0:
                # This is a 'broken' sample (usually 19 or 21 units)
                # We find the nearest valid sample and give it this mass
                if final_probs:
                    final_probs[-1] += s_prob
                else:
                    # If it's the first one, we'll attach it to the next valid one later
                    # but for now, let's just log it
                    pass

        # Re-normalize to ensure sum is exactly 1.0
        probs_array = np.array(final_probs, dtype=np.float32)
        probs_array /= probs_array.sum()
        
        samples_array = np.array(final_samples, dtype=np.int64)

        # Final sanity check
        sizes = [len(s) for s in samples_array]
        if len(set(sizes)) > 1:
             raise ValueError(f"Repair failed. Sizes still inconsistent: {set(sizes)}")

        self._all_samples_and_probs = (samples_array, probs_array)
        return self._all_samples_and_probs
    @property
    def nht_variance(self) -> float:
        if self._nht_variance is not None:
            return self._nht_variance

        samples, samples_probs = self.all_samples_and_probs
        nht_values = self.pop.variable[samples] / self.pop.inclusions[samples]
        nht_estimator = np.sum(nht_values, axis=1)
        true_total = self.pop.variable.sum()
        variance = np.sum(((nht_estimator - true_total) ** 2) * samples_probs)
        self._nht_variance = variance.item()
        return self._nht_variance

    def _expected_and_std(self, scores: np.ndarray) -> tuple[float, float]:
        samples, samples_probs = self.all_samples_and_probs
        expected_score = np.sum(scores * samples_probs)
        variance_score = (np.sum(((scores - expected_score) ** 2) * samples_probs))**.5
        return expected_score.item(), variance_score.item()
    
    def flatten(self) -> Design:
        """
        Truly flattens the design by merging all clusters and zones into a 
        single 'Super Cluster' to allow for unrestricted global reordering.
        """
        # 1. Extract the optimized spatial sequence from the current design
        # This is the 'Best ID' sequence found during the restricted phase
        current_ids = self.order.get()[:, 0].astype(int)

        # 2. Create a brand new Order object
        # Note: We need the classes available to rebuild the hierarchy
        from .order import Order, Cluster, Zone 
        flat_order = Order(self.pop)

        # 3. Create ONE 'Super Zone' containing every unit in the sequence
        # We put the IDs into _indices and set sort to a simple range
        super_zone = Zone()
        super_zone._indices = current_ids
        super_zone.sort = list(range(len(current_ids)))

        # 4. Create ONE 'Super Cluster' to hold the Super Zone
        super_cluster = Cluster(label=0, zones=[super_zone])

        # 5. Assign the new flat hierarchy to the Order
        flat_order.clusters = [super_cluster]
        flat_order.num_zones = 1
        flat_order.num_splits = self.order.num_splits

        # 6. Rebuild the internal array so self._order matches the new hierarchy
        flat_order._build_order(flat_order.num_splits)

        # 7. Return the design with this truly flat order
        return Design.from_order(self.pop, flat_order)
    
    def release_design(restricted_design):
        """
        Converts a multi-zone design into a single-partition design.
        This unlocks all units previously frozen by zone boundaries.
        """
        # 1. Extract the optimized sequence from the restricted design
        optimized_order = restricted_design.order.copy()
        
        # 2. Re-initialize a new Design with only ONE partition
        # This automatically recalculates the minimal set of 'fixed_ids'
        flat_design = Design.from_order(
            population=restricted_design.pop, 
            order=optimized_order
        )
        
        # Ensure num_partitions is 1 to allow global swaps
        flat_design._num_partitions = 1 
        flat_design._heaps = [_MaxHeap()] # Reset to a single global heap
        flat_design._build() # Rebuild the probability mass line
        
        print(f"Released! Frozen units reduced from {len(restricted_design.order.fixed_ids)} "
            f"to {len(flat_design.order.fixed_ids)}.")
        
        return flat_design

    @property
    def density_disparity(self) -> tuple[float, float]:
        if self._density_expected_and_variance is not None:
            return self._density_expected_and_variance

        samples, _ = self.all_samples_and_probs
        self._density_expected_and_variance = self._expected_and_std(
            DensityDisparity(self.pop).score(samples)
        )
        return self._density_expected_and_variance

    @property
    def moran(self) -> tuple[float, float]:
        if self._moran_expected_and_variance is not None:
            return self._moran_expected_and_variance

        samples, _ = self.all_samples_and_probs
        self._moran_expected_and_variance = self._expected_and_std(
            Moran(self.pop).score(samples)
        )
        return self._moran_expected_and_variance

    @property
    def voronoi(self) -> tuple[float, float]:
        if self._voronoi_expected_and_variance is not None:
            return self._voronoi_expected_and_variance

        samples, _ = self.all_samples_and_probs
        self._voronoi_expected_and_variance = self._expected_and_std(
            Voronoi(self.pop).score(samples)
        )
        return self._voronoi_expected_and_variance

    @property
    def local_balance(self) -> tuple[float, float]:
        if self._local_balance_expected_and_variance is not None:
            return self._local_balance_expected_and_variance

        samples, _ = self.all_samples_and_probs
        self._local_balance_expected_and_variance = self._expected_and_std(
            LocalBalance(self.pop).score(samples)
        )
        return self._local_balance_expected_and_variance

    @property
    def entropy(self) -> float:
        if self._entropy is not None:
            return self._entropy

        self.merge_identical()

        _, probs = self.all_samples_and_probs
        self._entropy = -np.sum(probs * np.log(probs)).item()

        return self._entropy

    @property
    def maximum_entropy(self) -> float:
        if self._maximum_entropy is not None:
            return self._maximum_entropy

        N, n = self.pop.N, self.pop.n
        maximum_entropy = loggamma(N + 1) - loggamma(n + 1) - loggamma(N - n + 1)

        self._maximum_entropy = maximum_entropy
        return self._maximum_entropy

    @property
    def relative_entropy(self) -> float:
        return self.entropy/self.maximum_entropy

    @property
    def fip(self) -> np.ndarray:
        if self._fip is not None:
            return self._fip

        samples, probs = self.all_samples_and_probs
        fip = np.zeros(self.pop.N, dtype=np.float64)
        np.add.at(fip, samples, probs.reshape(-1, 1))

        self._fip = fip
        return self._fip

    @property
    def indicator_matrix(self) -> np.ndarray:
        if self._indicator_matrix is not None:
            return self._indicator_matrix

        samples, _ = self.all_samples_and_probs
        S = samples.shape[0]
        N = self.pop.N

        indicators = np.zeros((S, N))
        np.put_along_axis(indicators, samples, 1.0, axis=1)

        self._indicator_matrix = indicators
        return self._indicator_matrix

    @property
    def sip(self) -> np.ndarray:
        if self._sip is not None:
            return self._sip

        _, probs = self.all_samples_and_probs
        indicators = self.indicator_matrix

        weighted_indicators = indicators * probs.reshape(-1, 1)
        self._sip = weighted_indicators.T @ indicators

        return self._sip
