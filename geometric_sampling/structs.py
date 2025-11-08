from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Iterator, Generic, Collection, Optional, Any
from typing import TypeVar

import numpy as np

from .type import ComparableNegatable

T = TypeVar("T", bound=ComparableNegatable)


class MaxHeap(Generic[T]):
    def __init__(
        self,
        initial_heap: Optional[Collection[T]] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.heap: list[T] = []
        if initial_heap is not None:
            self.heap = [-item for item in initial_heap]
            heapq.heapify(self.heap)
        self.rng = rng

    def push(self, item: T):
        heapq.heappush(self.heap, -item)

    def pop(self) -> T:
        return -heapq.heappop(self.heap)

    def peek(self) -> T:
        return -self.heap[0]

    def is_empty(self) -> bool:
        return len(self.heap) == 0

    def randompop(self) -> T:
        if not self.heap:
            raise IndexError("randompop from empty heap")

        items = [-neg for neg in self.heap]
        weights = [getattr(it, "probability", it) for it in items]
        total = sum(weights)

        if total <= 0:
            return self.pop()

        r = self.rng.random() * total

        cum = 0.0
        for idx, w in enumerate(weights):
            cum += w
            if cum >= r:
                chosen = items[idx]
                last = self.heap.pop()
                if idx < len(self.heap):
                    self.heap[idx] = last
                    heapq._siftup(self.heap, idx)
                    heapq._siftdown(self.heap, 0, idx)
                return chosen

        return self.pop()

    def copy(self) -> MaxHeap[T]:
        new_heap = MaxHeap[T](rng=self.rng)
        new_heap.heap = self.heap[:]
        new_heap.rng = self.rng
        return new_heap

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __iter__(self) -> Iterator[T]:
        return map(lambda x: -x, self.heap)

    def __str__(self):
        return str(list(map(lambda x: -x, self.heap)))

    def __hash__(self) -> int:
        return hash(tuple(self.heap))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaxHeap):
            return NotImplemented
        return self.heap == other.heap


@dataclass(order=False)
class Sample:
    probability: float
    ids: frozenset[int]
    index: list[int | list[int]] = None

    def almost_zero(self) -> bool:
        return self.probability < 1e-9

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sample):
            return NotImplemented
        return self.probability == other.probability and self.ids == other.ids

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Sample):
            return NotImplemented
        return self.probability < other.probability

    def __neg__(self) -> Sample:
        return Sample(-self.probability, self.ids, self.index)

    def __hash__(self):
        return hash(self.ids)


@dataclass(order=False)
class SampleGenetic:
    probability: float
    ids: frozenset[int]
    index: list[int | list[int]] = None

    def almost_zero(self) -> bool:
        return self.probability < 1e-9

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SampleGenetic):
            return NotImplemented
        return self.probability == other.probability and self.ids == other.ids

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SampleGenetic):
            return NotImplemented
        return self.probability < other.probability

    def __neg__(self) -> SampleGenetic:
        return SampleGenetic(-self.probability, self.ids, self.index)

    def __hash__(self):
        return hash(self.ids)
