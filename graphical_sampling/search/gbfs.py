import heapq
from itertools import chain, count
from typing import Generator, Callable, Literal, Any
import random
import pickle
from joblib import Parallel, delayed
from ..design import Design
from ..criteria import Criteria
from ..criteria.multi_objective import nht_variance_for_variable
import numpy as np
# =========================
# Swap debugger (window-gap)
# =========================
def swap_debugger(order, id_i, id_j, local_i, local_j, c_idx, z_idx):
    gap = abs(local_i - local_j)

    if not hasattr(order, "_last_swap_info"):
        order._last_swap_info = []

    order._last_swap_info.append({
        "id_i": int(id_i),
        "id_j": int(id_j),
        "gap": int(gap),
        "cluster": int(c_idx),
        "zone": int(z_idx)
    })

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
        # ==========================================
        # 🛡️ THE IMMORTAL RESERVOIR 🛡️
        # ==========================================
        self.reservoir_design = best_initial
        self.reservoir_val = self.best_criteria_value

    # All your static methods remain identical
    @staticmethod
    def _get(x: Any | Callable, iteration: int) -> Any:
        if isinstance(x, Callable):
            return x(iteration)
        return x

    @staticmethod
    def _apply_exchange(design, num_zones, num_changes, pull_strategy, exchange_coef):
        new_design = design.copy()
        for _ in range(num_changes):
            new_design.exchange(partitions=num_zones, pull_strategy=pull_strategy, 
                                exchange_coef=exchange_coef)
        return new_design

    @staticmethod
    def _apply_order_change(design, num_clusters, num_zones, num_changes, num_zone_changes, window):
        
        new_order = design.order.copy()   # 🔥 THIS WAS MISSING

        new_order._last_swap_info = []

        new_order.change(
            num_clusters, num_zones, num_changes, num_zone_changes,
            window=window,
            debugger_func=swap_debugger
        )

        new_design = design.from_order(design.pop, new_order)
        new_design._swap_info = new_order._last_swap_info

        return new_design
   
    def _order_neighbors(self, design, num_new_nodes, num_clusters, num_zones, num_changes, num_zone_changes, window) -> Generator:
        for _ in range(num_new_nodes):
            yield 'order_change', self._apply_order_change(design, num_clusters, num_zones, num_changes, num_zone_changes, window)

    def _exchange_neighbors(self, design, num_new_nodes, num_zones, num_changes, pull_strategy, exchange_coef, window) -> Generator:
        for _ in range(num_new_nodes):
            yield 'exchange', self._apply_exchange(design, num_zones, num_changes, pull_strategy, exchange_coef)

    def _update_top_k(self, design, criteria_value, k):
        heapq.heappush(self.top_k, (-criteria_value, next(self._counter), design))
        if len(self.top_k) > k:
            heapq.heappop(self.top_k)

    def run(
            self,
            max_iterations,
            max_open_set_size,
            top_k,
            num_new_order_nodes,
            num_new_exchange_nodes,
            num_clusters_range,
            num_zones_range,
            num_changes_range,
            num_zone_changes,
            pull_strategy='default',
            exchange_coef=0.75,
            num_explore=1,
            window=None,
            n_jobs=-1,
            backtrack_depth=50,
            stagnation_limit=10,
            output_details=False,
            y_main=None,
            y_aux=None,
        ):
        # --- SHOCK / BACKTRACK SETTINGS --
        stalls = 0             

        # --- RESUME LOGIC ---
        if not self.open_set:
            print("Starting fresh: Initializing Open Set.")
            for design in self.initial_designs:
                val = self.criteria(design)
                heapq.heappush(self.open_set, (val, next(self._counter), 'order_change', design))
                self._update_top_k(design, val, top_k)
        else:
            print(f"Resuming search with {len(self.open_set)} candidates in memory.")

        print(f"--- Parallel GBFS: Batch Size={num_explore}, Workers={n_jobs} ---")
        new_best = 'Nothingyet'
        num_improvements = 0
        num_improvements_reservoir = 0
        self.reservoir_val = self.best_criteria_value
        self.reservoir_design = self.best_design
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

            # =========================================================
            # 1. BATCH EXTRACTION & BACKTRACKING
            # =========================================================
            nodes_to_explore = []
            
            if stalls >= stagnation_limit:
                num_improvements = 0
                skip_depth = random.randint(backtrack_depth, len(self.open_set))
                print(f"\n⚡ STUCK. Purging the top {skip_depth} dead-ends to force a new path! ⚡")
                
                # 1. Permanently discard the nodes that got us stuck
                for _ in range(skip_depth):
                    if self.open_set:
                        heapq.heappop(self.open_set)
                
                # 2. Grab the target node from the new frontier
                while self.open_set and len(nodes_to_explore) < num_explore:
                    node = heapq.heappop(self.open_set)
                    if node[3] not in self.closed_set:
                        nodes_to_explore.append(node)
                
                # 3. COMMIT TO THE NEW PATH
                if nodes_to_explore:
                    new_start_val = nodes_to_explore[0][0]
                    self.best_criteria_value = new_start_val
                    self.best_design = nodes_to_explore[0][3]
                    print(f"Adopted new working baseline: {new_start_val:.6f}. Ready to improve little by little!\n")
                
                stalls = 0
                # ---> ADD THESE 4 LINES TO FIX THE JUPYTER MEMORY ERROR <---
                    
            else:
                # Normal Greedy Extraction
                while self.open_set and len(nodes_to_explore) < num_explore:
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

            if not unique_neighbors: 
                stalls += 1 
                continue
                
            designs_to_eval = list(unique_neighbors.keys())

            # 4. Parallel Evaluation
            criteria_values = Parallel(n_jobs=n_jobs)(delayed(self.criteria)(d) for d in designs_to_eval)

            # 5. Process Results
            improved_this_iter = False
            
            for new_design, new_val in zip(designs_to_eval, criteria_values):
                new_type = unique_neighbors[new_design]
                act_c = len(new_design.order.clusters)
                act_z = len(new_design.order.clusters[0].zones) if act_c > 0 else 0
                if output_details:
                    print(
                            f"ITR {iteration:4d} | "
                            f"IMP {num_improvements}-{num_improvements_reservoir} | "
                            # f"current_window: {current_window} | "
                            f"RSV {self.reservoir_val:.6f} | "
                            f"BST {self.best_criteria_value:.6f} [{new_best[:3].upper():3s}] | "
                            f"NEW {new_val:.6f} [{new_type[:3].upper():3s}] | "
                            f"{act_c:2d}C x {act_z:2d}Z | "
                            f"SIZ {len(new_design.all_samples_and_probs[1]):3d} | "
                            f"ENT {new_design.entropy:.4f}"
                        )
                # ==========================================
                # 🛡️ 1. CHECK THE IMMORTAL RESERVOIR
                # ==========================================
                # if round(new_val, 9) < round(self.reservoir_val, 9):
                #     num_improvements_reservoir += 1
                #     self.reservoir_val = new_val
                #     self.reservoir_design = new_design
                #     print(f"🏆 WOWW! RESERVOIR UPDATED! Absolute Best: {new_val:.6f} 🏆")
                #     print(
                #         f"ITR {iteration:4d} | "
                #         f"IMP {num_improvements:3d}-{num_improvements_reservoir:3d} | "
                #         f"RSV {self.reservoir_val:.6f} | "
                #         f"BST {self.best_criteria_value:.6f} [{new_best[:3].upper():3s}] | "
                #         f"NEW {new_val:.6f} [{new_type[:3].upper():3s}] | "
                #         f"{act_c:2d}C x {act_z:2d}Z | "
                #         f"SIZ {len(new_design.all_samples_and_probs[1]):3d} | "
                #         f"ENT {new_design.entropy:.4f}"
                #     )
                
                # ==========================================
                # 🚶 2. CHECK THE CURRENT WORKING PATH
                # ==========================================
                if round(new_val, 9) < round(self.best_criteria_value, 9):
                    if hasattr(new_design, "_swap_info"):
                        gaps = [s["gap"] for s in new_design._swap_info]
                        print(f"    GAP USED: {gaps}")
                    num_improvements += 1
                    improved_this_iter = True
                    new_best = new_type
                    var_main = (
                        nht_variance_for_variable(new_design, y_main)
                        if y_main is not None
                        else np.nan
                    )

                    var_aux = (
                        nht_variance_for_variable(new_design, y_aux)
                        if y_aux is not None
                        else new_design.nht_variance
                    )

                    mi_val = new_design.moran[0]
                    
                    # Clean print statement ONLY when we find an improvement
                    print(
                            f"ITR/SuccR {iteration:4d}/{num_improvements/(iteration+1):.3f} | "
                            f"OBJ {new_val:.6f} | "
                            f"VAR_main {var_main/10000:.6f} | "
                            f"VAR_aux {var_aux/10000:.6f} | "
                            f"MI {mi_val:.5f} | "
                            f"{act_c:2d}C x {act_z:2d}Z | "
                            f"SIZ {len(new_design.all_samples_and_probs[1]):3d} | "
                            f"ENT {new_design.entropy:.4f}"
                        )
                    
                    # Update local baseline
                    self.best_design = new_design
                    self.best_criteria_value = new_val

                self._update_top_k(new_design, new_val, top_k)
                heapq.heappush(self.open_set, (new_val, next(self._counter), new_type, new_design))

            # Update Stalls
            if improved_this_iter:
                stalls = 0
            else:
                stalls += 1

            # Pruning
            if len(self.open_set) > max_open_set_size * 2:
                self.open_set = heapq.nsmallest(max_open_set_size, self.open_set)
                heapq.heapify(self.open_set)
                
        # =========================================================
        # END OF RUN: RESTORE THE ABSOLUTE BEST FROM RESERVOIR
        # =========================================================
        self.best_design = self.reservoir_design
        self.best_criteria_value = self.reservoir_val
        print(f"--- Search Complete. Final All-Time Best (from Reservoir): {self.reservoir_val:.6f} ---")