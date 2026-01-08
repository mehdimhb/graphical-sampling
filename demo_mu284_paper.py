
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geometric_sampling.genetic_algorithm import GeometricSamplingGA


def load_mu284_paper_setup(sample_size: int = 15):
    """
    Load MU284
    """
    filename = "MU284_filtered.csv"

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: '{filename}' not found.")


    target_col = 'RMT85'
    aux_col = 'SS82'
    eval_col = 'CS82'

    x = df[aux_col].values
    y = df[target_col].values
    z = df[eval_col].values

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
    print(f"   Variable x (Auxiliary for pi): {aux_col}")
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


def plot_paper_correlations(df):
    """Replicates Figure 7 from the paper: Scatter matrix of variables."""
    print("\nGenerating Figure 7 reproduction...")
    subset = df[['RMT85', 'SS82', 'CS82']].rename(columns={
        'RMT85': 'y (RMT85)',
        'SS82': 'x (SS82)',
        'CS82': 'z (CS82)'
    })
    sns.pairplot(subset, kind="reg", diag_kind="kde")
    plt.suptitle("Replication of Paper Fig 7 (Outliers Removed)", y=1.02)
    plt.show()


def main():
    print("=" * 70)
    print("MU284 Experiment - Paper Section 6.2")
    print("=" * 70)

    # Step 1: Load Data
    try:
        data = load_mu284_paper_setup(sample_size=5)
    except Exception as e:
        print(e)
        return

    # Step 2: Run Genetic Algorithm
    print(f"\nStarting Genetic Optimization (n={data['n']})...")


    ga = GeometricSamplingGA(
        inclusions=data['inclusions'],
        auxiliary_var=data['eval_var'],
        main_var=data['target_var'],
        population_size=10,          # Increased population for better exploration
        elitism_rate=0.10,            # Keep top 10% unchanged
        mutation_intensity=15,         # Higher mutation intensity
        mutation_rate=0.10,            # Higher initial mutation rate
        use_partitions=True,
        adaptive_parameters=True,
        random_pull=True,             # Random pull for more exploration
        enable_monitoring=False,
        enable_live_plots=False,
        # NEW improved parameters
        crossover_rate=0.85,          # High crossover rate
        local_search_intensity=10,    # More local search iterations
        tournament_size=5,            # Larger tournament for selection
        restart_threshold=40,         # Restart diversity sooner
        diversity_threshold=0.02,     # Higher diversity threshold
    )

    best_design = ga.run(max_generations=1000, verbose=True)

    # Step 3: Final Report
    if ga.monitor and getattr(ga.monitor, "metrics_history", None):
        final = ga.monitor.metrics_history[-1]
        print(f"\nFinal Best Fitness: {final.best_fitness:.6f}")
        try:
            ga.monitor.generate_final_report(save_path="../mu284_paper_results.png")
            best_design.show()

            print("✅ Results plot saved to ../mu284_paper_results.png")
        except:
            pass


if __name__ == "__main__":
    main()