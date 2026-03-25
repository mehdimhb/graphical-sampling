
import os
import numpy as np
import pandas as pd
# from tqdm import tqdm

# Core project imports
# from graphical_sampling.sampling import KMeansSampler
from graphical_sampling.population import Population
from package_sampling.utils import inclusion_probabilities

from graphical_sampling.index import DensityDisparity
from graphical_sampling.index import Moran
from graphical_sampling.index import Voronoi
from graphical_sampling.index import LocalBalance


from graphical_sampling.index import Moran_r
from graphical_sampling.index import Voronoi_r
from graphical_sampling.index import LocalBalance_r

# B
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter

# Define the converter context for rpy2
combined_converter = ro.default_converter + numpy2ri.converter + pandas2ri.converter

def run_sampling_design(method, coords, probs, n, num_samples):
    N = len(coords)

    # 1. Python Methods (Nmcs and Rand)
    if method == "Nmcs":
        # Create the population and sampler
        pop = Population(coords=coords, probs=probs)
        sampler = KMeansSampler(
            population=pop,
            n=n,
            n_zones=(3, 3),
            zone_builder="sweep",
            units_order="spiral",
            zones_order="spiral"
        )
        # Return samples for simulation AND the sampler for math
        return sampler.sample(num_samples), sampler

    if method == "Rand":
        samples_idx = np.zeros((num_samples, n), dtype=int)
        for i in range(num_samples):
            samples_idx[i] = np.random.choice(N, n, replace=False) #
        return samples_idx, None

    # 2. R Methods (Lopi, Wave, Maxe, Scps)
    samples_idx = np.zeros((num_samples, n), dtype=int)

    with localconverter(combined_converter):
        ro.globalenv['coords_r'] = coords
        ro.globalenv['probs_r'] = probs

        ro.r("library(BalancedSampling)")
        ro.r("library(WaveSampling)")
        ro.r("library(sampling)")

        for i in range(num_samples):
            if method == "Lopi":
                samples_idx[i] = np.array(ro.r("lpm1(probs_r, coords_r)")) - 1 #
            # elif method == "Lopi2":
            #     samples_idx[i] = np.array(ro.r("lpm2(probs_r, coords_r)")) - 1
            elif method == "Scps":
                samples_idx[i] = np.array(ro.r("scps(probs_r, coords_r)")) - 1 #
            elif method == "Wave":
                mask = ro.r("wave(coords_r, probs_r)") #
                samples_idx[i] = np.where(np.array(mask).astype(bool))[0] #
            elif method == "Maxe":
                mask = ro.r("sampling::UPmaxentropy(probs_r)") #
                samples_idx[i] = np.where(np.array(mask).astype(bool))[0] #

    return samples_idx, None

import os
import warnings
import numpy as np
import pandas as pd
import gc

INCLUDE_NMCS_IN_SIM = False
FIND_PIK_EXPLOSION = False
sample_cnt = 5000

# pop_names = ['meuse']
pop_names = ['grid_N144_perm', ]#, 'clust_N144_perm']
print("baleee")
n_sizes = [32,]

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

base_dir = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(base_dir, "simulations", "populations", "simulated")
os.makedirs(data_folder, exist_ok=True)  # only if needed


results_folder = "/config/ws/graphical-sampling/simulations/results"
os.makedirs(results_folder, exist_ok=True)

inclusion = "UP"

# (Make sure pop_names and n_sizes are defined in your notebook!)
# Example: pop_names = ["meuse", "swiss", "rand"]
# Example: n_sizes = [10]
for name in pop_names:
    for n_size in n_sizes:
        n = n_size
        # 2. Load and Prep Data
        file_path = os.path.join(data_folder, f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        # df = df[0:30]
        coords = df[["x", "y"]].values.astype(float)

        # Assign target variable and inclusion probabilities
        if name == 'meuse':
            y_values = df["cadmium"].values
            pik = inclusion_probabilities(df["copper"].values, n_size)
        elif name == 'swiss':
            y_values = df["AREA_A"].values
            y_values = y_values.clip(5,100)
            p_values = df["AREA"].values.clip(5, 100)
            pik = inclusion_probabilities(p_values, n_size)
            # print("y_value",y_values)
            # print("p_value",p_values)
        else:
            y_values = df['z.90'].values
            pik = inclusion_probabilities(df["prob"].values, n_size)

        N = len(df)
        true_sum_y = np.sum(y_values)
        if inclusion == "EP":
            pik = inclusion_probabilities(np.ones(N), n_size)
            rho = 0
        else:
            rho = np.corrcoef(y_values, pik)[0, 1]

        # # 1. Force probabilities slightly away from absolute 0 and 1
        # pik = np.clip(pik, 1e-7, 1 - 1e-7)

        # # 2. Force the array to sum exactly to 'n' to fix floating-point drift
        # pik = pik * (n / np.sum(pik))

        print(f"\n--- Processing {name} (N={N}, n={n}, True Total={true_sum_y:.3f}, Rho={rho:.3f}) ---")

        # --- DEBUG BLOCK: Theoretical SRS Benchmark ---
        S2_y = np.var(y_values, ddof=1)
        theoretical_srs_var = (N**2) * (1 - n/N) * (S2_y / n)
        print(f"DEBUG: Theoretical SRS Total Variance: {theoretical_srs_var:.2f}")

        # 3. Setup Metrics (One-time initialization)
        pop_wrapped = Population(coords=coords, inclusions=pik)
        scorer_D = DensityDisparity(population=pop_wrapped)
        scorer_M = Moran(population=pop_wrapped, method='tille')
        scorer_V = Voronoi(population=pop_wrapped)
        scorer_L = LocalBalance(population=pop_wrapped)

        # 4. Sampling and Scoring Loop
        all_data = []
        # methods = ["Wave", "Lopi", "Scps", "Rand", "Maxe",  ]
        methods = ["Lopi"]
        # methods = ["Wave"]
        if INCLUDE_NMCS_IN_SIM:
            methods.insert(0, "Nmcs")

        for m in methods:
            print(f"Running simulation for {m}...")

            samples, _ = run_sampling_design(m, coords, pik, n, sample_cnt)
            # Vectorized scoring (High Efficiency)
            d_scores = scorer_D.score(samples)
            m_scores = scorer_M.score(samples)
            v_scores = scorer_V.score(samples)
            l_scores = scorer_L.score(samples)

            # Calculate Estimators
            if m == "Rand":
                est_scores = N * np.mean(y_values[samples], axis=1)
            else:
                est_scores = np.sum(y_values[samples] / pik[samples], axis=1)

            # Store results for EVERY iteration
            for i in range(sample_cnt):
                if m == 'Wave': # & (i + 1) %  == 0):
                    print(f"    {m}: {i+1} / {sample_cnt} iterations completed")
                all_data.append([N, n, sample_cnt, rho, m, d_scores[i], v_scores[i], m_scores[i], l_scores[i], est_scores[i]])

                # --- TRACKING RARE EXTREME ESTIMATES ---
                if FIND_PIK_EXPLOSION:
                    error_margin = abs(est_scores[i] - true_sum_y) / true_sum_y
                    if error_margin > 0.5:
                        print(f"\n[ALERT] Method: {m} | Iteration: {i}")
                        print(f"Estimate: {est_scores[i]:.2f} | True Total: {true_sum_y:.2f} | Error: {error_margin:.2%}")

                        bad_sample_idx = samples[i]
                        bad_y = y_values[bad_sample_idx]
                        bad_pik = pik[bad_sample_idx]

                        contributions = bad_y / bad_pik
                        culprit_local_idx = np.argmax(contributions)
                        culprit_pop_idx = bad_sample_idx[culprit_local_idx]

                        print(f"Sample Indices: {bad_sample_idx}")
                        print(f"Exploded Unit Index: {culprit_pop_idx} | y: {bad_y[culprit_local_idx]} | pik: {bad_pik[culprit_local_idx]}")

        # 5. Data Processing & Aggregation
        res_df = pd.DataFrame(all_data, columns=["N", "n", "ite" ,"rho", "Method", "D", "V", "M", "L", "HT"])

        # Aggregate simulation results (ignores N, n, rho because we group by Method only on target cols)
        summary = res_df.groupby("Method").agg({
            "D": ["mean", "std"],
            "V": ["mean", "std"],
            "M": ["mean", "std"],
            "L": ["mean", "std"],
            "HT": ["mean", "var"]
        })

        # CRITICAL: Force column order before renaming
        summary = summary[["D", "V", "M", "L", "HT"]]
        summary.columns = ["Dm", "Ds", "Vm", "Vs", "Mm", "Ms", "Lm", "Ls", "HTm", "HTv"]

        # 6. Inject Theoretical Nmcs row (If available)
        theoretical_results = None
        if theoretical_results:
            t = theoretical_results
            summary.loc["Nmcs"] = [
                t['Exp_Density'], t['SD_Density'],
                t['Exp_Voronoi'], t['SD_Voronoi'],
                t['Exp_Moran'], t['SD_Moran'],
                t['Exp_Local'], t['SD_Local'],
                true_sum_y, t['HT_Variance']
            ]

        # 7. Final Metrics: RB and Efficiency (Eff)
        summary["RB"] = (summary["HTm"] - true_sum_y) / true_sum_y
        if "Rand" in summary.index:
            rand_var = summary.loc["Rand", "HTv"]
            summary["Eff"] = rand_var / summary["HTv"].replace(0, np.nan)
        else:
            summary["Eff"] = np.nan

        summary['rho'] = rho
        summary['ite'] = sample_cnt
        summary['n'] = n
        summary['N'] = N
        # Final Table Output formatting
        final_cols = ['N', 'n', 'ite','rho', "HTm", "HTv" ,"Eff", "RB", "Dm", "Ds", "Vm", "Vs", "Mm", "Ms", "Lm", "Ls"]
        summary = summary.reindex(columns=final_cols)
        print(summary.round(3))

        # 8. Save Data
        # A. Save individual summary
        summary_file = os.path.join(results_folder, f"summary_{name}_n={n}_EUP={inclusion}.csv")
        summary.to_csv(summary_file)

        # B. Save extended raw iterations (Append Mode)
        # extended_file = os.path.join(results_folder, "all_iterations_extended.csv")
        # res_df.to_csv(extended_file, mode='a', index=False, header=not os.path.exists(extended_file))

print("\nAll simulations completed and saved!")