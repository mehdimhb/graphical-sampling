import numpy as np
#_____________________________________________________________________________
class MCTSNode:
    def __init__(self, design, parent=None):
        self.design = design
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward_sum = 0.0

    def add_child(self, child):
        self.children.append(child)

#_____________________________________________________________________________
class MCTS:

    def __init__(
        self,
        initial_design,
        criteria,
        switch_coefficient=0.5,
        random_pull=False,
        base_changes=10,
        delta_increase=1,
        delta_decrease=1,
    ):
        """
        MCTS Option C (Adaptive Mutation) + Reward Option 1 (Symmetric Normalized)
        """

        self.root = MCTSNode(initial_design)
        self.criteria = criteria
        self.switch_coefficient = switch_coefficient
        self.random_pull = random_pull

        self.base_changes = base_changes
        self.delta_increase = delta_increase
        self.delta_decrease = delta_decrease

        self.initial_criteria_value = criteria(initial_design)

        self.best_criteria_value = self.initial_criteria_value
        self.best_design = initial_design.copy()

        print("=== MCTS INITIALIZED ===")
        print("Initial Var:", self.initial_criteria_value)
        print("------------------------")
#_____________________________________________________________________________
# Calculate depth of node

    def _get_depth(self, node):
        depth = 0
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth
#_____________________________________________________________________________
# Selection

    def _select_child(self, node, C=1.4):
        best_score = -1e18
        best_child = None

        N = max(1, node.visits)

        for child in node.children:
            if child.visits == 0:
                return child

            mean_reward = child.reward_sum / child.visits
            explore = np.sqrt(np.log(N) / child.visits)
            score = mean_reward + C * explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child
#_____________________________________________________________________________
# Expansion

    def _expand(self, node, rollout_improved):
        if rollout_improved:
            mut = max(1, self.base_changes - self.delta_decrease)
        else:
            mut = self.base_changes + self.delta_increase

        print(
            "  EXPAND | depth =", self._get_depth(node),
            "| rollout_improved =", rollout_improved,
            "| mutations =", mut
        )
        new_design = node.design.copy()
        for _ in range(mut):
            new_design.iterate(
                random_pull=self.random_pull,
                switch_coefficient=self.switch_coefficient,
            )

        child = MCTSNode(new_design, parent=node)
        node.add_child(child)
        return child
    #_____________________________________________________________________________
    # Simulation

    def _simulate(self, design, rollout_depth):
        cur = design.copy()

        parent_var = self.criteria(cur)
        best_var = parent_var

        print("  SIMULATE | start Var =", parent_var)

        for _ in range(rollout_depth):
            cur.iterate(
                random_pull=self.random_pull,
                switch_coefficient=self.switch_coefficient,
            )
            v = self.criteria(cur)
            if v < best_var:
                best_var = v

        reward = (self.initial_criteria_value - best_var) / (
            self.initial_criteria_value + best_var
        )

        print(
            "  ROLLOUT RESULT | best Var =", best_var,
            "| reward =", reward
        )

        if best_var < self.best_criteria_value:
            print(
                "  >>> GLOBAL BEST UPDATE:",
                self.best_criteria_value,
                "→",
                best_var
            )
            self.best_criteria_value = best_var
            self.best_design = cur.copy()

        improved = best_var < parent_var

        return reward, improved
#_____________________________________________________________________________
# Backpropagation

    def _backprop(self, node, reward):
        depth = self._get_depth(node)
        print(
            "  BACKPROP | reward =", reward,
            "| from depth =", depth
        )

        while node is not None:
            node.visits += 1
            node.reward_sum += reward
            node = node.parent
#_____________________________________________________________________________
# Run Monte Carlo Tree Search Algorithm

    def run(
        self,
        max_iterations=200,
        max_children_per_node=20,
        rollout_depth=100,
    ):
        rollout_improved = False

        print("=== MCTS RUN STARTED ===")
        print("Max iterations:", max_iterations)
        print("------------------------")

        for it in range(max_iterations):

            print(f"\n>>> ITERATION {it}")
            node = self.root
            
            while len(node.children)!= 0 :
                node = self._select_child(node)
            
                

            print(
                "  SELECTED NODE | depth =",
                self._get_depth(node),
                "| visits =",
                node.visits,
                "| children =",
                len(node.children)
            )

            while node.visits >= 0 and len(node.children) < max_children_per_node:
                child = self._expand(node, rollout_improved)
                print('length of node.children: ', len(node.children) )
                reward, rollout_improved = self._simulate(
                child.design, rollout_depth
                )
                self._backprop(child, reward)



            print(
                "  ITER END | best Var so far =",
                self.best_criteria_value
            )

        print("\n=== MCTS FINISHED ===")
        print("Best Var:", self.best_criteria_value)

        return self.best_design, self.best_criteria_value
 #_____________________________________________________________________________               