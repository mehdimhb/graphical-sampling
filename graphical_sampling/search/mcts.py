import os

# -----------------------------------------------------------------------------
# CPU LOCK (Speed Guarantee)
# -----------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
import time
import numpy as np
from typing import List, Union


# =============================================================================
# WORKER: ADAPTIVE BINARY (Big Step vs Small Step)
# =============================================================================
def worker_adaptive_binary(args):
    (design_copy, criteria, rollout_depth, initial_criteria_value, 
     random_pull, switch_coefficient, seed) = args
    
    np.random.seed(seed % (2**32 - 1))
    
    current_design = design_copy
    current_val = criteria(current_design)
    
    best_var = current_val
    best_design_local = None 

    ROLLOUT_STEPS = 18
    
    for _ in range(ROLLOUT_STEPS):
        
        agg_coeff = min(1.0, switch_coefficient * 1.3)
        
        cand_a = current_design.copy()
        cand_a.iterate(random_pull=random_pull, switch_coefficient=agg_coeff)
        val_a = criteria(cand_a)
        
        accepted_cand = None
        accepted_val = float('inf')
        
        if val_a < current_val:
            accepted_cand = cand_a
            accepted_val = val_a
        else:
            cons_coeff = max(0.01, switch_coefficient * 0.7)
            
            cand_b = current_design.copy()
            cand_b.iterate(random_pull=random_pull, switch_coefficient=cons_coeff)
            val_b = criteria(cand_b)
            
            if val_a < val_b:
                accepted_cand = cand_a
                accepted_val = val_a
            else:
                accepted_cand = cand_b
                accepted_val = val_b
        
        current_design = accepted_cand
        current_val = accepted_val
        
        if current_val < best_var:
            best_var = current_val
            best_design_local = current_design.copy()
            
        # قطع امید (Safety Cutoff)
        if current_val > initial_criteria_value * 1.15:
            break

    if best_design_local is None:
        best_design_local = design_copy 
        
    if best_var < initial_criteria_value:
        improvement = (initial_criteria_value - best_var) / initial_criteria_value
        reward = improvement * 100 
    else:
        reward = 0.0
    
    return reward, best_var, best_design_local

# =============================================================================
# MCTS CLASSES
# =============================================================================

class MCTSNode:
    def __init__(self, design, parent=None, mutation_magnitude=0, mutation_type='root'):
        self.design = design
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward_sum = 0.0
        self.mutation_magnitude = mutation_magnitude
        self.mutation_type = mutation_type 

    def add_child(self, child):
        self.children.append(child)
        
    @property
    def value(self):
        return self.reward_sum / self.visits if self.visits > 0 else 0.0


class MCTS:
    def __init__(self, initial_designs: Union[List, object], criteria, switch_coefficient=0.5, random_pull=False, base_changes=10, min_changes=1, exploration_constant=0.1):
        """
        Modified MCTS to handle multiple initial designs (Population-based MCTS).
        Args:
            initial_designs: A single Design object OR a list of Design objects.
        """
        if not isinstance(initial_designs, list):
            self.initial_designs = [initial_designs]
        else:
            self.initial_designs = initial_designs

        self.criteria = criteria
        self.switch_coefficient = switch_coefficient
        self.random_pull = random_pull
        self.base_changes = base_changes
        self.min_changes = min_changes
        self.C = exploration_constant
        
        self.best_criteria_value = float('inf')
        self.best_design = None
        
        print(f"=== MULTI-START MCTS (Population Size: {len(self.initial_designs)}) ===")
        

        for d in self.initial_designs:
            val = criteria(d)
            if val < self.best_criteria_value:
                self.best_criteria_value = val
                self.best_design = d.copy()

        self.initial_criteria_value = self.best_criteria_value
        
        self.root = MCTSNode(design=None, mutation_type='virtual_root')
        
        for i, d in enumerate(self.initial_designs):
            child_node = MCTSNode(d, parent=self.root, mutation_type=f'init_pop_{i}')
            self.root.add_child(child_node)

        self.last_improvement_iter = 0
        self.teleport_count = 0
        print(f"Best Initial Criteria: {self.best_criteria_value}")

    def _get_depth(self, node):
        depth = 0
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        N = max(1, node.visits)
        for child in node.children:
            if child.visits == 0: return child
            
            mean_reward = child.value
            explore = np.sqrt(np.log(N) / child.visits)
            score = mean_reward + self.C * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _create_child_node(self, parent_node, num_mutations, m_type):
        if parent_node.design is None:
            raise ValueError("Attempted to expand Virtual Root via mutation. This shouldn't happen.")
            
        new_design = parent_node.design.copy()
        for _ in range(num_mutations):
            new_design.iterate(random_pull=self.random_pull, switch_coefficient=self.switch_coefficient)
        child = MCTSNode(new_design, parent=parent_node, mutation_magnitude=num_mutations, mutation_type=m_type)
        return child

    def _calculate_mutation_distribution(self, node, total_children_needed):
        quality = node.value
        if node.visits == 0 and node.parent is not None and node.parent.design is not None: 
            quality = node.parent.value
        
        if quality > 0.5: 
            low_count = int(total_children_needed * 0.5)
            mid_count = int(total_children_needed * 0.2)
            high_count = total_children_needed - low_count - mid_count
        else:
            high_count = int(total_children_needed * 0.5)
            mid_count = int(total_children_needed * 0.3)
            low_count = total_children_needed - high_count - mid_count
        return high_count, mid_count, low_count
    
    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward_sum += reward
            node = node.parent

    def run(self, max_iterations=150, max_children_per_node=8):
        print("=== RUN STARTED (ADAPTIVE POPULATION MODE) ===")
        
        cpu_count = multiprocessing.cpu_count()
        self.last_improvement_iter = 0
        floor_high_mut = max(4, int(self.base_changes * 0.4))
        
        with multiprocessing.Pool(processes=cpu_count) as pool:
            
            for it in range(max_iterations):
                
                if (it - self.last_improvement_iter) > 3:
                    self.teleport_count += 1
                    print(f"\n  >>> STAGNATION (8 iters) -> TELEPORT #{self.teleport_count} to Best ({self.best_criteria_value:.2f})")
                    self.root = MCTSNode(self.best_design.copy(), mutation_type=f'teleport_{self.teleport_count}')
                    self.last_improvement_iter = it 
                    self.initial_criteria_value = self.best_criteria_value

                node = self.root
                
                # --- TRAVERSAL PHASE ---
                
                if node.mutation_type == 'virtual_root':
                    node = self._select_child(node)
                
                while len(node.children) > 0 and len(node.children) >= max_children_per_node: 
                     node = self._select_child(node)

                # --- EXPANSION PHASE ---
                needed = max_children_per_node - len(node.children)
                children_to_simulate = []
                sim_args_list = []
                
                if needed > 0:
                    depth = self._get_depth(node)
                    decay = depth // 3
                    curr_high_mag = max(floor_high_mut, self.base_changes)
                    curr_mid_mag = max(2, (self.base_changes // 2) - decay)
                    curr_low_mag = self.min_changes
                    
                    n_high, n_mid, n_low = self._calculate_mutation_distribution(node, needed)
                    
                    if it % 20 == 0:
                        print(f"\n>>> ITER {it} | Depth {depth} | Val: {node.value:.2f}")

                    configs = [
                        (n_high, curr_high_mag, 'high'),
                        (n_mid, curr_mid_mag,   'mid'),
                        (n_low, curr_low_mag,   'low')
                    ]
                    
                    for count, mag, m_type in configs:
                        for _ in range(count):
                            child = self._create_child_node(node, mag, m_type)
                            node.add_child(child)
                            child.rollout_depth_config = 0 
                            children_to_simulate.append(child)

                    for i, child in enumerate(children_to_simulate):
                        seed = int(time.time() * 1000) + id(child) + i
                        args = (child.design.copy(), self.criteria, 0, self.initial_criteria_value, self.random_pull, self.switch_coefficient, seed)
                        sim_args_list.append(args)
                
                if not children_to_simulate: continue
                
                # --- SIMULATION (ROLLOUT) PHASE ---
                results = pool.map(worker_adaptive_binary, sim_args_list)
                
                # --- BACKPROPAGATION PHASE ---
                for child, res in zip(children_to_simulate, results):
                    reward, best_val_in_rollout, best_design_in_rollout = res
                    
                    if best_val_in_rollout < self.best_criteria_value:
                        imp = ((self.best_criteria_value - best_val_in_rollout) / self.best_criteria_value) * 100
                        if imp > 0.0001:
                            print(f"  ★ BEST: {self.best_criteria_value:.1f} -> {best_val_in_rollout:.1f} (Type: {child.mutation_type})")
                        self.best_criteria_value = best_val_in_rollout
                        self.best_design = best_design_in_rollout
                        self.last_improvement_iter = it
                        
                    self._backpropagate(child, reward)

        print(f"\n=== FINISHED ===\nBest: {self.best_criteria_value}")
        return self.best_design, self.best_criteria_value