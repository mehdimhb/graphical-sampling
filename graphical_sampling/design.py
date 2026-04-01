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

    def _build(self):
        events: list[tuple[float, str, int]] = []
        level: float = 0

        for idx, share in self.order.get():
            p = self.pop.inclusions[int(idx)] * share
            next_level = level + p
            if next_level < 1 - 1e-9:
                events.append((level, "start", idx))
                events.append((next_level, "end", idx))
                level = next_level
            elif next_level > 1 + 1e-9:
                events.append((level, "start", idx))
                events.append((1, "end", idx))
                events.append((0, "start", idx))
                events.append((next_level - 1, "end", idx))
                level = next_level - 1
            else:
                events.append((level, "start", idx))
                events.append((1, "end", idx))
                level = 0

        for i in range(self.num_partitions + 1):
            events.append((i / self.num_partitions, "boundary", -1))

        events.sort()
        active = set()
        last_point: float = 0

        self.events = events

        for point, event_type, bar_index in events:
            if point > last_point + 1e-9:
                if active:
                    midpoint = (last_point + point) / 2
                    zone_idx = min(int(midpoint * self.num_partitions), self.num_partitions - 1)

                    length = round(point - last_point, 9)
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
            exchange_coef: float = 0.75
    ) -> None:
        if not (0 < partitions <= self.num_partitions):
            raise ValueError(
                f"partitions must be greater than 0 and less than or equal to {self.num_partitions}. Got {partitions}."
            )

        valid_partitions = [i for i, h in enumerate(self._heaps) if len(h) >= 2]
        if not valid_partitions:
            return

        actual_count = min(partitions, len(valid_partitions))
        selected_partitions = self._rng.choice(valid_partitions, size=actual_count, replace=False)

        for part_idx in selected_partitions:
            part_idx = int(part_idx)

            sample1 = self._pull(part_idx, 'largest' if pull_strategy == 'default' else pull_strategy)
            sample2 = self._pull(part_idx, 'random' if pull_strategy == 'default' else pull_strategy)

            if sample1.ids == sample2.ids:
                self._push(part_idx, _Sample(sample1.prob + sample2.prob, sample1.ids))
            else:
                self._push(part_idx, *self._switch(sample1, sample2, exchange_coef))

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

        all_samples = []
        all_prob = []

        for sample in self:
            all_samples.append(list(sample.ids))
            all_prob.append(sample.prob)

        samples_array = np.array(all_samples, dtype=np.int64)
        probs_array = np.array(all_prob, dtype=np.float32)

        if probs_array.size > 0:
            probs_array *= 1.0 / probs_array.sum()

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
