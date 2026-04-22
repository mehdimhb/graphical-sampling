import heapq
import random
from itertools import chain, count
from typing import Generator, Callable, Literal, Any
from collections import deque
from joblib import Parallel, delayed

# FIX: Relative import changed to look in the parent directory
from ..index import Moran 
from ..design import Design
from ..criteria import Criteria

PullStrategy = Literal['default', 'random', 'largest']

class GreedyBestFirstSearchTabu:
    def __init__(self, initial_designs: list[Design], criteria: Criteria) -> None:
        self.initial_designs = initial_designs
        self.criteria = criteria
        self._counter = count()
        self.top_k: list[tuple[float, int, Design]] = []  
        
        best_initial = min(self.initial_designs, key=self.criteria)
        self.best_design = best_initial
        self.best_criteria_value = self.criteria(best_initial)

    @staticmethod
    def _get(x: Any | Callable, iteration: int) -> Any:
        if isinstance(x, Callable):
            return x(iteration)
        return x

    @staticmethod
    def _apply_exchange(design, num_zones, num_changes, pull_strategy, exchange_coef, window):
        new_design = design.copy()
        for _ in range(num_changes):
            # We pass the window here
            new_design.exchange(partitions=num_zones, pull_strategy=pull_strategy, 
                            exchange_coef=exchange_coef, window=window)
        return new_design

    @staticmethod
    def _apply_order_change(design, num_clusters, num_zones, num_changes, num_zone_changes, window):
        new_order = design.order.copy()
        # We pass the window here
        new_order.change(num_clusters, num_zones, num_changes, num_zone_changes, window=window)
        return design.from_order(design.pop, new_order)

    def _order_neighbors(
            self, design: Design, num_new_nodes: int, num_clusters: int, num_zones: int, num_changes: int,
            num_zone_changes: int,
            window: int | None = None # <--- Added window here
    ) -> Generator[tuple[str, Design], None, None]:
        for _ in range(num_new_nodes):
            yield 'order_change', self._apply_order_change(
                design, num_clusters, num_zones, num_changes, num_zone_changes, window
            )

    def _exchange_neighbors(
            self, design: Design, num_new_nodes: int, num_zones: int,
            num_changes: int, pull_strategy: str, exchange_coef: float,
            window: int | None = None # <--- Added window here
    ) -> Generator[tuple[str, Design], None, None]:
        for _ in range(num_new_nodes):
            yield 'exchange', self._apply_exchange(
                design, num_zones, num_changes, pull_strategy, exchange_coef, window
            )

    def _update_top_k(self, design: Design, criteria_value: float, k: int) -> None:
        heapq.heappush(self.top_k, (-criteria_value, next(self._counter), design))
        if len(self.top_k) > k:
            heapq.heappop(self.top_k)

    def get_top_k_results(self) -> list[tuple[float, Design]]:
        sorted_heap = sorted(self.top_k, key=lambda x: x[0], reverse=True)
        return [(-val, design) for val, _, design in sorted_heap]

    def run(
            self,
            max_iterations: int,
            max_open_set_size: int,
            top_k: int,
            num_new_order_nodes: int | Callable[[int], int],
            num_new_exchange_nodes: int | Callable[[int], int],
            num_clusters_range: tuple[int, int],
            num_zones_range: tuple[int, int],
            num_changes_range: tuple[int, int],
            num_zone_changes: int | Callable[[int], int],
            pull_strategy: PullStrategy = 'default',
            exchange_coef: float = 0.75,
            num_explore: int = 1,
            window: int | Callable[[int], int] | None = None,
            n_jobs: int = -1,
            tabu_size: int = 10000 
    ) -> None:
        closed_set = set()
        tabu_list = deque(maxlen=tabu_size)
        open_set = []

        # Warm-up pre-caches the W matrix for all parallel workers
        if isinstance(self.criteria, Moran):
            _ = self.criteria.score(self.initial_designs[0].all_samples_and_probs[0][:1])

        log_interval = max(1, max_iterations // 10)

        for design in self.initial_designs:
            val = self.criteria(design)
            heapq.heappush(open_set, (val, next(self._counter), 'order_change', design))
            self._update_top_k(design, val, top_k)

        for iteration in range(max_iterations):
            current_window = self._get(window, iteration)
            if not open_set:
                break

            cooling_factor = 1.0 - (iteration / max_iterations)
            
            def cooled_range(r):
                low, high = r
                new_high = max(low + 1, int(low + (high - low) * cooling_factor))
                return random.randint(low, new_high)

            num_changes = cooled_range(num_changes_range)
            num_clusters = cooled_range(num_clusters_range)
            num_zones = cooled_range(num_zones_range)

            if iteration % log_interval == 0:
                print(f"Iter {iteration:5d} | Best: {self.best_criteria_value:.4f} | Open: {len(open_set):5d}")

            nodes_to_explore = []
            for _ in range(num_explore):
                if not open_set: break
                nodes_to_explore.append(heapq.heappop(open_set))

            all_neighbors = []
            for val, _, node_type, current_design in nodes_to_explore:
                if current_design in closed_set:
                    continue

                closed_set.add(current_design)
                tabu_list.append(current_design)

                neighbors = self._exchange_neighbors(
                    current_design,
                    self._get(num_new_exchange_nodes, iteration),
                    num_zones,
                    num_changes,
                    self._get(pull_strategy, iteration),
                    exchange_coef,
                    current_window 
                )

                if node_type == 'order_change':
                    order_neighbors = self._order_neighbors(
                        current_design,
                        self._get(num_new_order_nodes, iteration),
                        num_clusters,
                        num_zones,
                        num_changes,
                        self._get(num_zone_changes, iteration),
                        current_window
                    )
                    neighbors = chain(neighbors, order_neighbors)

                all_neighbors.extend(neighbors)

            unique_neighbors = {}
            for new_type, new_design in all_neighbors:
                if new_design not in closed_set and new_design not in unique_neighbors:
                    unique_neighbors[new_design] = new_type

            if not unique_neighbors:
                continue

            designs_to_eval = list(unique_neighbors.keys())
            criteria_values = Parallel(n_jobs=n_jobs)(
                delayed(self.criteria)(design) for design in designs_to_eval
            )

            for new_design, new_val in zip(designs_to_eval, criteria_values):
                new_type = unique_neighbors[new_design]
                if new_val < self.best_criteria_value:
                    act_c = len(new_design.order.clusters)
                    # We look at the first cluster to see the number of zones per cluster
                    act_z = len(new_design.order.clusters[0].zones) if act_c > 0 else 0
                    
                    print(
                        f"Update @ {iteration:4d}: {new_val:.6f} | "
                        f"Type: {new_type:12s} | "
                        # f"Scope: C={num_clusters}, Z={num_zones} | " # Search params
                        f"Actual: {act_c}Cx{act_z}Z | "             # Internal structure
                        # f"Partitions: {new_design.num_partitions}"  # Total heaps
                    )
                    self.best_design = new_design
                    self.best_criteria_value = new_val

                self._update_top_k(new_design, new_val, top_k)
                heapq.heappush(open_set, (new_val, next(self._counter), new_type, new_design))

            if len(open_set) > max_open_set_size * 2:
                unique_open_nodes = {}
                while open_set and len(unique_open_nodes) < max_open_set_size:
                    node = heapq.heappop(open_set)
                    design_obj = node[3]
                    if design_obj not in unique_open_nodes:
                        unique_open_nodes[design_obj] = node
                
                open_set = list(unique_open_nodes.values())
                heapq.heapify(open_set)

        print(f"--- Search Complete. Final Best Value: {self.best_criteria_value:.4f} ---")