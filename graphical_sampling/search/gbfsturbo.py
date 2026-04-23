import heapq
from itertools import chain, count
from typing import Generator, Callable, Literal, Any
import random
from joblib import Parallel, delayed

from ..design import Design
from ..criteria import Criteria


PullStrategy = Literal['default', 'random', 'largest']


class GreedyBestFirstSearchTurbo:
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

    def debug_neighbor_generation(design):
        print("--- Debugging Neighbor Generation ---")
        new_order = design.order.copy()
        
        # Test 1: Can the order be copied?
        print(f"Original ID: {id(design.order)}, Copy ID: {id(new_order)}")
        
        # Test 2: Does .change() actually modify the sequence?
        # Force a change: 1 cluster, 2 zones, 0 unit swaps, 1 zone shuffle
        new_order.change(num_clusters=1, num_zones=2, num_changes=0, num_zone_changes=1)
        
        orig_seq = list(design.order._order if hasattr(design.order, '_order') else design.order.order)
        new_seq = list(new_order._order if hasattr(new_order, '_order') else new_order.order)
        
        if orig_seq == new_seq:
            print("❌ ERROR: .change() produced an identical sequence!")
            print("Check if num_zones_range > 1 and if your clusters actually have zones.")
        else:
            print("✅ SUCCESS: .change() modified the sequence.")
        
        # Test 3: Design creation
        try:
            new_design = design.from_order(design.pop, new_order)
            print(f"✅ SUCCESS: New design created with type {type(new_design)}")
        except Exception as e:
            print(f"❌ ERROR: design.from_order failed: {e}")

    # Call this with your initial best design
    # debug_neighbor_generation(initial_design)


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
            window: int | Callable[[int], int] | None = None,
            n_jobs: int = -1
    ) -> None:
        closed_set = set()
        open_set = []  # Min-heap of (criteria_value, counter, type_str, design)

        print(f"--- Starting Parallel GBFS: Iterations={max_iterations}, Batch={num_explore}, Workers={n_jobs} ---")

        # Initial baseline
        for design in self.initial_designs:
            val = self.criteria(design)
            heapq.heappush(open_set, (val, next(self._counter), 'order_change', design))
            self._update_top_k(design, val, top_k)

        for iteration in range(max_iterations):
            current_window = self._get(window, iteration)
            
            # Smart Memory Management
            if iteration % 1000 == 0 and iteration > 0:
                # Keep only the designs we are actually looking at to prevent memory bloating
                current_active = {node[3] for node in nodes_to_explore} if 'nodes_to_explore' in locals() else set()
                closed_set = current_active 

            if not open_set: break

            # 1. Pop the best nodes
            nodes_to_explore = []
            for _ in range(num_explore):
                if not open_set: break
                node = heapq.heappop(open_set)
                if node[3] not in closed_set:
                    nodes_to_explore.append(node)

            if not nodes_to_explore: continue

            # 2. Fast Sequential Neighbor Generation
            all_neighbors = []
            
            # Pre-calculate ranges for this iteration
            n_order = self._get(num_new_order_nodes, iteration)
            n_exch = self._get(num_new_exchange_nodes, iteration)
            n_changes = random.randint(num_changes_range[0], num_changes_range[1])
            n_clusters = random.randint(num_clusters_range[0], num_clusters_range[1])
            n_zones = random.randint(num_zones_range[0], num_zones_range[1])

            for val, _, node_type, current_design in nodes_to_explore:
                closed_set.add(current_design)

                # Generate Exchange Neighbors
                all_neighbors.extend(self._exchange_neighbors(
                    current_design, n_exch, n_zones, n_changes, 
                    self._get(pull_strategy, iteration), exchange_coef, current_window
                ))

                # Generate Order Neighbors
                if node_type == 'order_change':
                    all_neighbors.extend(self._order_neighbors(
                        current_design, n_order, n_clusters, n_zones, 
                        n_changes, self._get(num_zone_changes, iteration), current_window
                    ))

            # 3. Clean Deduplication (Done once, correctly)
            unique_neighbors = {}
            for n_type, n_design in all_neighbors:
                if n_design not in closed_set and n_design not in unique_neighbors:
                    unique_neighbors[n_design] = n_type

            if not unique_neighbors: continue

            designs_to_eval = list(unique_neighbors.keys())

            # 4. Parallel Criteria Evaluation (The actual heavy lifting)
            criteria_values = Parallel(n_jobs=n_jobs)(
                delayed(self.criteria)(d) for d in designs_to_eval
            )

            # 5. Process and Update
            for new_design, new_val in zip(designs_to_eval, criteria_values):
                new_type = unique_neighbors[new_design]
                
                if new_val < self.best_criteria_value:
                    act_c = len(new_design.order.clusters)
                    act_z = len(new_design.order.clusters[0].zones) if act_c > 0 else 0
                    print(f"Update @ {iteration:4d}: {new_val:.6f} | {new_type:12s} | "
                          f"DSize: {len(new_design.all_samples_and_probs[1]):4d} | Entp: {new_design.entropy:.4f}")
                    
                    self.best_design = new_design
                    self.best_criteria_value = new_val

                self._update_top_k(new_design, new_val, top_k)
                heapq.heappush(open_set, (new_val, next(self._counter), new_type, new_design))

            # 6. Keep the open set manageable
            if len(open_set) > max_open_set_size * 2:
                open_set = heapq.nsmallest(max_open_set_size, open_set)
                heapq.heapify(open_set)

        print(f"--- Search Complete. Final Best Value: {self.best_criteria_value:.4f} ---")