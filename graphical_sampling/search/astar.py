import heapq
import random
from dataclasses import dataclass
from itertools import count
from typing import Generator, Any, List, Tuple

from ..design import Design
from ..red_black_tree import RedBlackTree


@dataclass(frozen=True, order=False, eq=False)
class Node:
    criteria_value: float
    design: Design

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.criteria_value < other.criteria_value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.criteria_value == other.criteria_value

    def __le__(self, other: Any) -> bool:
        return self < other or self == other

    def __ge__(self, other: Any) -> bool:
        return not self < other

    def __gt__(self, other: Any) -> bool:
        return other < self


class AStar:
    def __init__(
            self,
            initial_designs: List[Design],
            *,
            switch_coefficient: float = 0.5,
            random_pull: bool = True,
    ) -> None:
        self.initial_designs = initial_designs
        self.switch_coefficient = switch_coefficient
        self.random_pull = random_pull

        # We use a counter to handle tie-breaking in the heap without comparing Designs directly
        self._counter = count()
        self.top_k = []  # Heap of (-value, tie_breaker, design)

        # Initialize the best stats based on the list of initial designs
        best_initial = min(self.initial_designs, key=lambda d: d.nht_variance)
        self.best_design = best_initial
        self.best_criteria_value = best_initial.nht_variance

    def iterate_design(self, design: Design, num_changes: int) -> Design:
        new_design = design.copy()
        for _ in range(num_changes):
            new_design.iterate(
                random_pull=self.random_pull,
                switch_coefficient=self.switch_coefficient,
            )
        return new_design

    def neighbors(
            self,
            design: Design,
            num_new_nodes: int,
            num_changes_interval: Tuple[int, int],
    ) -> Generator[Design, None, None]:
        for _ in range(num_new_nodes):
            # Choose random number of changes from the interval [min, max]
            changes = random.randint(*num_changes_interval)
            yield self.iterate_design(design, changes)

    def _update_top_k(self, design: Design, criteria_value: float, k: int):
        """
        Maintains the top K best designs (lowest criteria_value) using a heap.
        We use a Max-Heap of size K (simulated by inverting values in a Min-Heap)
        to efficiently evict the worst of the current best.
        """
        # Push negative value to simulate Max-Heap behavior with heapq (Min-Heap)
        entry = (-criteria_value, next(self._counter), design)
        heapq.heappush(self.top_k, entry)

        # If we have more than K items, pop the one with the largest criteria_value
        # (which corresponds to the smallest negative value in the heap)
        if len(self.top_k) > k:
            heapq.heappop(self.top_k)

    def get_top_k_results(self) -> List[Tuple[float, Design]]:
        """Returns the sorted list of top K designs (positive values)."""
        # Sort by best (lowest variance) first
        # We stored them as negative, so we sort descending to get -1, -2, -3...
        # then invert back to positive.
        sorted_heap = sorted(self.top_k, key=lambda x: x[0], reverse=True)
        return [(-val, design) for val, _, design in sorted_heap]

    def run(
            self,
            max_iterations: int,
            num_new_nodes: int,
            max_open_set_size: int,
            num_changes: Tuple[int, int],
            top_k: int,
    ):
        closed_set = set()
        open_set = RedBlackTree[Node]()

        # Initialize Open Set and Top K with all initial designs
        for design in self.initial_designs:
            val = design.nht_variance
            open_set.insert(Node(val, design))
            self._update_top_k(design, val, top_k)

        for it in range(max_iterations):
            if not open_set:
                break

            mn = open_set.get_min()
            if not mn:
                break

            current_design = mn.design
            if current_design in closed_set:
                continue

            closed_set.add(current_design)

            for new_design in self.neighbors(
                    current_design, num_new_nodes, num_changes
            ):
                new_criteria_value = new_design.nht_variance

                if new_design in closed_set:
                    continue

                # Update Open Set (RedBlackTree)
                if len(open_set) < max_open_set_size:
                    open_set.insert(Node(new_criteria_value, new_design))
                else:
                    mx = open_set.get_max()
                    # Only insert if better than the worst in the open set
                    if mx is None or mx.criteria_value > new_criteria_value:
                        if mx is not None:
                            open_set.remove(mx)
                        open_set.insert(Node(new_criteria_value, new_design))

                # Update Global Best
                if new_criteria_value < self.best_criteria_value:
                    self.best_design = new_design
                    self.best_criteria_value = new_criteria_value

                # Update Top K Heap
                self._update_top_k(new_design, new_criteria_value, top_k)
