from dataclasses import dataclass
from typing import Generator, Any
import bisect
import numpy as np
from joblib import Parallel, delayed

from ..criteria.criteria import Criteria
from ..new_design import NewDesign
from ..red_black_tree import RedBlackTree

@dataclass(frozen=True, order=False, eq=False)
class Node:
    criteria_value: float
    design: NewDesign

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
        initial_designs: list[NewDesign],
        criteria: Criteria,
        *,
        threshold: float = -1.0,
    ) -> None:
        self.initial_designs = initial_designs
        self.criteria = criteria
        self.threshold = threshold
        self.rng = np.random.default_rng()

        # Evaluate initial designs
        self.initial_criteria_value = np.array([
            self.criteria(design) for design in self.initial_designs
        ])
        self.best_design = self.initial_designs[np.argmax(self.initial_criteria_value)]
        self.best_criteria_value = np.min(self.initial_criteria_value)

        print('best initial criteria value', self.best_criteria_value)

    def iterate_design(
        self,
        design: NewDesign,
        n_clusters_to_change_order_zone: int | str = 'all',
        n_clusters_to_change_order_units: int | str = 'all',
        n_zones_to_change_order_units: int | str = 'all',
        n_changes_in_order_of_units: int = 1,
        n_changes_in_order_of_zones: int = 1
    ) -> NewDesign:
        new_design = design.copy()
        new_design.iterate(
            n_clusters_to_change_order_zone,
            n_clusters_to_change_order_units,
            n_zones_to_change_order_units,
            n_changes_in_order_of_units,
            n_changes_in_order_of_zones,
        )
        return new_design

    def run(
        self,
        max_iterations: int,
        num_new_nodes: int,
        max_open_set_size: int,
        n_clusters_to_change_order_zone: int | str = 'all',
        n_clusters_to_change_order_units: int | str = 'all',
        n_zones_to_change_order_units: int | str = 'all',
        n_changes_in_order_of_units: int = 1,
        n_changes_in_order_of_zones: int = 1,
        n_jobs: int = -1,
    ):
        # Initialize open set as a sorted list
        open_set_list: list[Node] = []
        for i, design in enumerate(self.initial_designs):
            bisect.insort_left(
                open_set_list,
                Node(float(self.initial_criteria_value[i]), design)
            )

        # Track visited designs to avoid revisiting
        closed_set: set[NewDesign] = set()

        for it in range(max_iterations):
            if not open_set_list:
                break

            current_node = open_set_list.pop(0)
            current_design = current_node.design

            # Skip already-visited designs
            if current_design in closed_set:
                continue
            closed_set.add(current_design)

            print(f"\nparent node: {current_node.criteria_value}")

            # Generate neighbors in parallel
            new_designs = Parallel(n_jobs=n_jobs)(
                delayed(self.iterate_design)(
                    current_design,
                    n_clusters_to_change_order_zone,
                    n_clusters_to_change_order_units,
                    n_zones_to_change_order_units,
                    n_changes_in_order_of_units,
                    n_changes_in_order_of_zones,
                ) for _ in range(num_new_nodes)
            )

            # Evaluate and insert each new node to maintain sorted order
            for new_design in new_designs:
                # Skip if neighbor already seen
                if new_design in closed_set:
                    continue

                new_criteria_value = self.criteria(new_design)
                new_node = Node(new_criteria_value, new_design)
                print(f"child node: {new_criteria_value}")

                if len(open_set_list) < max_open_set_size:
                    bisect.insort_left(open_set_list, new_node)
                else:
                    worst = open_set_list[-1]
                    if new_node.criteria_value < worst.criteria_value:
                        open_set_list.pop()
                        bisect.insort_left(open_set_list, new_node)

                # Update global best
                if new_criteria_value < self.best_criteria_value:
                    self.best_design = new_design
                    self.best_criteria_value = new_criteria_value
                    print('\n==============================================')
                    print(f'New best criteria value: {self.best_criteria_value}\n')
                    if self.best_criteria_value < self.threshold:
                        return it

        return max_iterations
