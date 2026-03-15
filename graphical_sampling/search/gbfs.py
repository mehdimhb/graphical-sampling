import heapq
from itertools import chain, count
from typing import Generator, Callable, Literal, Any
import random
from joblib import Parallel, delayed

from ..design import Design
from ..criteria import Criteria


PullStrategy = Literal['default', 'random', 'largest']


class GreedyBestFirstSearch:
    def __init__(self, initial_designs: list[Design], criteria: Criteria) -> None:
        self.initial_designs = initial_designs
        self.criteria = criteria

        # Counter handles tie-breaking in the heap to prevent comparing Design objects
        self._counter = count()
        self.top_k: list[tuple[float, int, Design]] = []  # Heap of (-value, tie_breaker, design)

        # Initialize global bests
        best_initial = min(self.initial_designs, key=self.criteria)
        self.best_design = best_initial
        self.best_criteria_value = self.criteria(best_initial)

    @staticmethod
    def _get(x: Any | Callable, iteration: int) -> Any:
        if isinstance(x, Callable):
            return x(iteration)
        return x

    @staticmethod
    def _apply_exchange(
            design: Design, num_zones: int, num_changes: int, pull_strategy: PullStrategy, exchange_coef: float
    ) -> Design:
        new_design = design.copy()
        for _ in range(num_changes):
            new_design.exchange(partitions=num_zones, pull_strategy=pull_strategy, exchange_coef=exchange_coef)
        return new_design

    @staticmethod
    def _apply_order_change(
            design: Design, num_clusters: int, num_zones: int, num_changes: int, num_zone_changes
    ) -> Design:
        new_order = design.order.copy()
        new_order.change(num_clusters, num_zones, num_changes, num_zone_changes)
        return design.from_order(design.pop, new_order)

    def _order_neighbors(
            self, design: Design, num_new_nodes: int, num_clusters: int, num_zones: int, num_changes: int,
            num_zone_changes: int
    ) -> Generator[tuple[str, Design], None, None]:
        for _ in range(num_new_nodes):
            yield 'order_change', self._apply_order_change(
                design, num_clusters, num_zones, num_changes, num_zone_changes
            )

    def _exchange_neighbors(
            self, design: Design, num_new_nodes: int, num_zones: int,
            num_changes: int, pull_strategy: PullStrategy, exchange_coef: float
    ) -> Generator[tuple[str, Design], None, None]:
        for _ in range(num_new_nodes):
            yield 'exchange', self._apply_exchange(design, num_zones, num_changes, pull_strategy, exchange_coef)

    def _update_top_k(self, design: Design, criteria_value: float, k: int) -> None:
        """Maintains the top K best designs using a Min-Heap containing negative values."""
        heapq.heappush(self.top_k, (-criteria_value, next(self._counter), design))
        if len(self.top_k) > k:
            heapq.heappop(self.top_k)

    def get_top_k_results(self) -> list[tuple[float, Design]]:
        """Returns the sorted list of top K designs."""
        sorted_heap = sorted(self.top_k, key=lambda x: x[0], reverse=True)
        return [(-val, design) for val, _, design in sorted_heap]


    # def _rand_or_fixed(arg, iteration):
    #     """If arg is a tuple like (min, max), return random int between them.
    #     If arg is callable or fixed number, handle accordingly."""
    #     if isinstance(arg, tuple) and len(arg) == 2:
    #         return random.randint(arg[0], arg[1])
    #     elif callable(arg):
    #         return arg(iteration)
    #     else:
    #         return arg


    def run(
            self,
            max_iterations: int,
            max_open_set_size: int,
            top_k: int,
            num_new_order_nodes: int | Callable[[int], int],
            num_new_exchange_nodes: int | Callable[[int], int],
            num_clusters_range: int | tuple[int, int],
            num_zones_range: int | tuple[int, int],
            num_changes_range: int | tuple[int, int],
            num_zone_changes: int | Callable[[int], int],
            pull_strategy: PullStrategy = 'default',
            exchange_coef: float = 0.75,
            num_explore: int = 1,
            n_jobs: int = -1
    ) -> None:
        closed_set = set()
        open_set = []  # Min-heap of (criteria_value, counter, type_str, design)

        log_interval = max(1, max_iterations // 10)
        print(
            f"--- Starting Parallel GBFS: Max Iterations={max_iterations}, Batch Size={num_explore}, Workers={n_jobs} ---")

        # Initialize Open Set and Top K
        for design in self.initial_designs:
            val = self.criteria(design)
            heapq.heappush(open_set, (val, next(self._counter), 'order_change', design))
            self._update_top_k(design, val, top_k)

        print(f"Initial best criteria value: {self.best_criteria_value:.4f}")

        for iteration in range(max_iterations):
            num_changes = random.randint(num_changes_range[0],num_changes_range[0])
            num_clusters = random.randint(num_clusters_range[0],num_clusters_range[0])
            num_zones = random.randint(num_zones_range[0],num_zones_range[0])
            if not open_set:
                print(f"Search exhausted at iteration {iteration}: Open set is empty.")
                break

            if iteration % log_interval == 0:
                print(
                    f"Iter {iteration:5d}/{max_iterations} | Best: {self.best_criteria_value:.4f} | Open: {len(open_set):5d} | Closed: {len(closed_set):5d}")

            # 1. Batch Extraction: Pop up to `num_explore` nodes
            nodes_to_explore = []
            for _ in range(num_explore):
                if not open_set:
                    break
                nodes_to_explore.append(heapq.heappop(open_set))

            all_neighbors = []
            valid_expansion = False

            # 2. Sequential Neighbor Generation
            # (Generating designs is usually fast; it's the criteria evaluation that is slow)
            for val, _, node_type, current_design in nodes_to_explore:
                if current_design in closed_set:
                    continue

                valid_expansion = True
                closed_set.add(current_design)

                neighbors = self._exchange_neighbors(
                    current_design,
                    self._get(num_new_exchange_nodes, iteration),
                    self._get(num_zones, iteration),
                    self._get(num_changes, iteration),
                    self._get(pull_strategy, iteration),
                    self._get(exchange_coef, iteration))

                if node_type == 'order_change':
                    order_neighbors = self._order_neighbors(
                        current_design,
                        self._get(num_new_order_nodes, iteration),
                        self._get(num_clusters, iteration),
                        self._get(num_zones, iteration),
                        self._get(num_changes, iteration),
                        self._get(num_zone_changes, iteration),
                    )
                    neighbors = chain(neighbors, order_neighbors)

                all_neighbors.extend(neighbors)

            # Skip the rest of the loop if all popped nodes were already closed
            if not valid_expansion:
                continue

            # 3. Deduplication: Prevent redundant evaluations in the same batch
            unique_neighbors = {}
            for new_type, new_design in all_neighbors:
                if new_design not in closed_set and new_design not in unique_neighbors:
                    unique_neighbors[new_design] = new_type

            if not unique_neighbors:
                continue

            designs_to_eval = list(unique_neighbors.keys())

            # 4. Parallel Criteria Evaluation
            criteria_values = Parallel(n_jobs=n_jobs)(
                delayed(self.criteria)(design) for design in designs_to_eval
            )

            # 5. Process Results
            for new_design, new_val in zip(designs_to_eval, criteria_values):
                new_type = unique_neighbors[new_design]
                # for new_design, new_val in zip(designs_to_eval, criteria_values):
                #     # new_type = unique_neighbors[new_design]

                #     if new_type == "order_change":
                #         p1 = sorted(current_design.all_samples_and_probs[1])
                #         p2 = sorted(new_design.all_samples_and_probs[1])

                #         print("\n--- DEBUG REORDER CHECK ---")
                #         print("Parent_probs   |   Child_probs")
                #         for a, b in zip(p1, p2):
                #             print(f"{a:.6f}   |   {b:.6f}")
                #             print(sum(p1), sum(p2))
                #         print("---------------------------\n")

                # Update Global Best
                if new_val < self.best_criteria_value:
                    print(
                        f"{iteration}: {new_val:.7f},  "
                        f"{new_type},  "
                        f"DSize:{len(new_design.all_samples_and_probs[0])},  "
                        f"var:{new_design.nht_variance:.2f}, "
                        f"Entp:{new_design.entropy:.2f}"
                        )
                    self.best_design = new_design
                    self.best_criteria_value = new_val

                # Update Top K Heap
                self._update_top_k(new_design, new_val, top_k)

                # Update Open Set
                heapq.heappush(open_set, (new_val, next(self._counter), new_type, new_design))

            # --- Lazy Pruning ---
            if len(open_set) > max_open_set_size * 2:
                open_set = heapq.nsmallest(max_open_set_size, open_set)
                heapq.heapify(open_set)

        print(f"--- Search Complete. Final Best Value: {self.best_criteria_value:.4f} ---")
