from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Iterator, Any

import numpy as np


@dataclass(frozen=True)
class _Sample:
    prob: float
    ids: frozenset[int]

    def almost_zero(self) -> bool:
        return self.prob < 1e-9

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _Sample):
            return NotImplemented
        return self.ids == other.ids

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, _Sample):
            return NotImplemented
        return self.prob < other.prob

    def __neg__(self) -> _Sample:
        return _Sample(-self.prob, self.ids)

    def __hash__(self) -> int:
        return hash(self.ids)


class _MaxHeap:
    def __init__(
        self,
        initial_heap: list[_Sample] | None = None
    ):
        self.heap: list[_Sample] = []
        if initial_heap is not None:
            self.heap = initial_heap
            heapq.heapify(self.heap)
        self.rng = np.random.default_rng()

    def push(self, item: _Sample):
        heapq.heappush(self.heap, -item)

    def pop(self) -> _Sample:
        return -heapq.heappop(self.heap)

    def peek(self) -> _Sample:
        return -self.heap[0]

    def random_pop(self) -> _Sample:
        idx = self.rng.integers(len(self.heap))
        val = -self.heap[idx]
        self.heap[idx] = self.heap[-1]
        self.heap.pop()
        if idx < len(self.heap):
            heapq._siftup(self.heap, idx)  # type: ignore
            heapq._siftdown(self.heap, 0, idx)  # type: ignore
        return val

    def copy(self) -> _MaxHeap:
        new_heap = _MaxHeap()
        new_heap.heap = self.heap[:]
        new_heap.rng = self.rng
        return new_heap

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __iter__(self) -> Iterator[_Sample]:
        return map(lambda x: -x, self.heap)

    def __str__(self):
        return str(list(map(lambda x: -x, self.heap)))

    def __hash__(self) -> int:
        return hash(tuple(self.heap))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaxHeap):
            return NotImplemented
        return self.heap == other.heap
