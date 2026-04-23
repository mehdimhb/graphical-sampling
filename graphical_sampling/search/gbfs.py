import heapq
from itertools import chain, count
from typing import Generator, Callable, Literal, Any
import random
import pickle
from joblib import Parallel, delayed

# No changes to your types or imports
PullStrategy = Literal['default', 'random', 'largest']

class GreedyBestFirstSearch:
    def __init__(self, initial_designs: list['Design'], criteria: 'Criteria') -> None:
        self.initial_designs = initial_designs
        self.criteria = criteria
        self._counter = count()
        self.top_k: list[tuple[float, int, 'Design']] = []

        # --- CORE PRESERVATION: State is now attached to self ---
        self.open_set = [] 
        self.closed_set = set()

        # Initialize global bests
        best_initial = min(self.initial_designs, key=self.criteria)
        self.best_design = best_initial
        self.best_criteria_value = self.criteria(best_initial)

    # All your static methods remain identical
    @staticmethod
    def _get(x: Any | Callable, iteration: int) -> Any:
        if isinstance(x, Callable):
            return x(iteration)
        return x

    @staticmethod
    def _apply_exchange(design, num_zones, num_changes, pull_strategy, exchange_coef, window):
        new_design = design.copy()
        for _ in range(num_changes):
            new_design.exchange(partitions=num_zones, pull_strategy=pull_strategy, 
                                exchange_coef=exchange_coef, window=window)
        return new_design

    @staticmethod
    def _apply_order_change(design, num_clusters, num_zones, num_changes, num_zone_changes, window):
        new_order = design.order.copy()
        new_order.change(num_clusters, num_zones, num_changes, num_zone_changes, window=window)
        return design.from_order(design.pop, new_order)
   
    def _order_neighbors(self, design, num_new_nodes, num_clusters, num_zones, num_changes, num_zone_changes, window) -> Generator:
        for _ in range(num_new_nodes):
            yield 'order_change', self._apply_order_change(design, num_clusters, num_zones, num_changes, num_zone_changes, window)

    def _exchange_neighbors(self, design, num_new_nodes, num_zones, num_changes, pull_strategy, exchange_coef, window) -> Generator:
        for _ in range(num_new_nodes):
            yield 'exchange', self._apply_exchange(design, num_zones, num_changes, pull_strategy, exchange_coef, window)

    def _update_top_k(self, design, criteria_value, k):
        heapq.heappush(self.top_k, (-criteria_value, next(self._counter), design))
        if len(self.top_k) > k:
            heapq.heappop(self.top_k)

    def run(self, max_iterations, max_open_set_size, top_k, num_new_order_nodes, num_new_exchange_nodes,
            num_clusters_range, num_zones_range, num_changes_range, num_zone_changes,
            pull_strategy='default', exchange_coef=0.75, num_explore=1, window=None, n_jobs=-1):

        # --- RESUME LOGIC ---
        # Only initialize if the open set is actually empty (first run)
        if not self.open_set:
            print("Starting fresh: Initializing Open Set.")
            for design in self.initial_designs:
                val = self.criteria(design)
                heapq.heappush(self.open_set, (val, next(self._counter), 'order_change', design))
                self._update_top_k(design, val, top_k)
        else:
            print(f"Resuming search with {len(self.open_set)} candidates in memory.")

        log_interval = max(1, max_iterations // 10)
        print(f"--- Parallel GBFS: Batch Size={num_explore}, Workers={n_jobs} ---")

        for iteration in range(max_iterations):
            current_window = self._get(window, iteration)
            
            # Maintenance
            if iteration % 1000 == 0 and iteration > 0:
                current_designs = {node[3] for node in nodes_to_explore} if 'nodes_to_explore' in locals() else set()
                self.closed_set = current_designs 

            num_changes = random.randint(num_changes_range[0], num_changes_range[1])
            num_clusters = random.randint(num_clusters_range[0], num_clusters_range[1])
            num_zones = random.randint(num_zones_range[0], num_zones_range[1])

            if not self.open_set:
                print(f"Search exhausted at iteration {iteration}.")
                break

            # 1. Batch Extraction
            nodes_to_explore = []
            for _ in range(num_explore):
                if not self.open_set: break
                node = heapq.heappop(self.open_set)
                if node[3] not in self.closed_set:
                    nodes_to_explore.append(node)

            if not nodes_to_explore: continue

            # 2. Sequential Neighbor Generation
            all_neighbors = []
            for val, _, node_type, current_design in nodes_to_explore:
                self.closed_set.add(current_design)

                neighbors = self._exchange_neighbors(
                    current_design, self._get(num_new_exchange_nodes, iteration),
                    num_zones, num_changes, self._get(pull_strategy, iteration), exchange_coef, current_window
                )

                if node_type == 'order_change':
                    order_neighbors = self._order_neighbors(
                        current_design, self._get(num_new_order_nodes, iteration),
                        num_clusters, num_zones, num_changes, self._get(num_zone_changes, iteration), current_window
                    )
                    neighbors = chain(neighbors, order_neighbors)

                all_neighbors.extend(neighbors)

            # 3. Deduplication
            unique_neighbors = {}
            for n_type, n_design in all_neighbors:
                if n_design not in self.closed_set and n_design not in unique_neighbors:
                    unique_neighbors[n_design] = n_type

            if not unique_neighbors: continue
            designs_to_eval = list(unique_neighbors.keys())

            # 4. Parallel Evaluation
            criteria_values = Parallel(n_jobs=n_jobs)(delayed(self.criteria)(d) for d in designs_to_eval)

            # 5. Process Results
            for new_design, new_val in zip(designs_to_eval, criteria_values):
                new_type = unique_neighbors[new_design]
                if new_val < self.best_criteria_value:
                    act_c = len(new_design.order.clusters)
                    act_z = len(new_design.order.clusters[0].zones) if act_c > 0 else 0
                    print(f"new@ {iteration:4d}: {new_val:.6f} | {new_type:12s} | {act_c}Cx{act_z}Z | "
                          f"DSize: {len(new_design.all_samples_and_probs[1]):4d} | "
                          f"Entp:{new_design.entropy:.4f}")
                    self.best_design = new_design
                    self.best_criteria_value = new_val

                self._update_top_k(new_design, new_val, top_k)
                heapq.heappush(self.open_set, (new_val, next(self._counter), new_type, new_design))

            # Pruning
            if len(self.open_set) > max_open_set_size * 2:
                self.open_set = heapq.nsmallest(max_open_set_size, self.open_set)
                heapq.heapify(self.open_set)

        print(f"--- Search Complete. Final Best: {self.best_criteria_value:.4f} ---")