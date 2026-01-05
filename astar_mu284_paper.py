
import numpy as np
import pandas as pd
import time

from geometric_sampling.design import Design
from geometric_sampling.search.astar import AStar
from geometric_sampling.criteria.var_nht import VarNHT


def load_mu284_paper_setup(sample_size: int = 5):

    filename = "MU284_filtered.csv"
    df = pd.read_csv(filename)

    target_col = 'RMT85'
    aux_col = 'SS82'
    eval_col = 'CS82'

    x = df[aux_col].values.astype(float)
    y = df[target_col].values.astype(float)
    z = df[eval_col].values.astype(float)

    total_x = np.sum(x)
    pi = (sample_size * x) / total_x

    # Check feasibility
    if np.any(pi > 1.0):
        max_feasible = int(total_x / np.max(x))
        print(f"WARNING: n={sample_size} leads to probs > 1. Max feasible n is {max_feasible}.")
        sample_size = max_feasible
        pi = (sample_size * x) / total_x

    # Print Stat Summary matching Paper Description
    print(f"\nPaper Setup (Section 6.2):")
    print(f"   Variable y (Target): {target_col}")
    print(f"   Variable x (Auxiliary for π): {aux_col}")
    print(f"   Variable z (Evaluation for OGFS): {eval_col}")
    print(f"   Correlation (x, y): {df[aux_col].corr(df[target_col]):.3f} (Paper: ~0.45)")
    print(f"   Correlation (z, y): {df[eval_col].corr(df[target_col]):.3f} (Paper: ~0.46)")

    return {
        'inclusions': pi,
        'auxiliary_var': x,
        'target_var': y,
        'eval_var': z,
        'n': sample_size,
        'data': df
    }


def main():
    print("=" * 70)
    print("A* Search on MU284 - Same Settings as demo_mu284_paper.py")
    print("=" * 70)

    # Step 1: Load Data with same sample_size=5 as demo_mu284_paper.py
    try:
        data = load_mu284_paper_setup(sample_size=5)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"\nPopulation size N: {len(data['inclusions'])}")
    print(f"Sample size n: {data['n']}")
    print(f"Sum of inclusion probs: {sum(data['inclusions']):.4f}")

    # Step 2: Create initial design
    print("\nCreating initial design...")
    initial_design = Design(inclusions=data['inclusions'])

    # Step 3: Create criteria (VarNHT using CS82 as evaluation variable - same as GA)
    criteria = VarNHT(
        auxiliary_variable=data['eval_var'],
        inclusion_probability=data['inclusions']
    )

    initial_fitness = criteria(initial_design)
    print(f"Initial VarNHT: {initial_fitness:.6f}")

    # Step 4: Run A* Search
    print(f"\nStarting A* Search Optimization (n={data['n']})...")
    print("=" * 50)

    astar = AStar(
        initial_design=initial_design,
        criteria=criteria,
        switch_coefficient=0.5,
        random_pull=True,
        threshold=1e-6,
    )

    # Run with similar effort to GA (100 generations * 30 population ≈ 3000 evaluations)
    # A* with 100 iterations * 100 nodes
    start_time = time.time()
    iterations_used = astar.run(
        max_iterations=100,
        num_new_nodes=100,
        max_open_set_size=100,
        num_changes=10  # Similar to mutation_intensity=10
    )
    elapsed_time = time.time() - start_time

    # Step 5: Results
    print("=" * 50)
    print(f"A* Search completed after {iterations_used} iterations")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"\nInitial VarNHT: {initial_fitness:.6f}")
    print(f"Final VarNHT: {astar.best_criteria_value:.6f}")
    improvement = (1 - astar.best_criteria_value/initial_fitness) * 100
    print(f"Improvement: {improvement:.2f}%")
    print(f"\nBest design contains {len(list(astar.best_design))} samples")

    # Validate the design
    def validate_design(design, inclusions):
        id_probs = {}
        for sample in design:
            for unit in sample.ids:
                id_probs[unit] = id_probs.get(unit, 0) + sample.probability

        for i, expected in enumerate(inclusions):
            actual = id_probs.get(i, 0)
            if abs(actual - expected) > 1e-6:
                return False
        return True

    is_valid = validate_design(astar.best_design, data['inclusions'])
    if is_valid:
        print("✅ Best design passes validation!")
    else:
        print("❌ Best design failed validation!")


if __name__ == "__main__":
    main()

