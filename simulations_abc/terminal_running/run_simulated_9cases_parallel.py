"""Run the simulated 9-case ABC experiment in parallel.

This mirrors the current notebook setup:
- z variables: z_90, z_80, z_00
- size variables for pi: size_90, size_80, size_00
- size_00 gives equal probabilities pi = n / N
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "simulations_abc" / "terminal_running" / "run_abc_mu284_terminal.py"

N_UNITS = 50
N_SAMPLE = 5
SIMULATED_CORRELATIONS = [0.90, 0.80, 0.00]
PI_SIZE_STRENGTH = 0.35
PI_MODE = "auxiliary"
PI_MIX_ALPHA = 1.0
OBJECTIVE = "min_relative"
INITIAL_OMEGA_VALUE = 0.0
INITIAL_RHO_VALUE = 0.5
BASE_SEED = 202600384
LIMIT = 5
ONLOOKER_FACTOR = 0.5


def load_runner():
    spec = importlib.util.spec_from_file_location("vincent_runner", RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(runner)
    return runner


def standardize(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    if sd == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def make_size_for_pi(x):
    return np.exp(PI_SIZE_STRENGTH * standardize(x))


def build_pi_from_size_variable(runner, size_values, size_name=None):
    if PI_MODE == "equal" or size_name == "size_00":
        return np.full(len(size_values), N_SAMPLE / len(size_values))

    size_aux = make_size_for_pi(size_values)
    if PI_MODE == "auxiliary":
        size = size_aux
    elif PI_MODE == "mix":
        size = (1.0 - PI_MIX_ALPHA) * np.mean(size_aux) + PI_MIX_ALPHA * size_aux
    else:
        raise ValueError("PI_MODE must be auxiliary, equal, or mix")

    return runner.inclusionprobabilities(size, N_SAMPLE)


def corr_suffix(corr):
    return f"{int(round(corr * 100)):02d}"


def correlated_with_y(y_std, corr, rng):
    noise = rng.normal(size=len(y_std))
    noise = noise - np.mean(noise)
    noise = noise - np.dot(noise, y_std) / np.dot(y_std, y_std) * y_std
    noise = standardize(noise)
    return corr * y_std + np.sqrt(max(0.0, 1.0 - corr**2)) * noise


def make_simulated_population(seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    location = rng.uniform(80.0, 120.0)
    scale = rng.uniform(8.0, 18.0)
    y = location + scale * rng.normal(size=N_UNITS)
    y_std = standardize(y)

    data = {"unit": np.arange(N_UNITS), "y": y}
    z_names = []
    size_names = []

    for corr in SIMULATED_CORRELATIONS:
        suffix = corr_suffix(corr)
        z_name = f"z_{suffix}"
        size_name = f"size_{suffix}"
        data[z_name] = correlated_with_y(y_std, corr, rng)
        data[size_name] = correlated_with_y(y_std, corr, rng)
        z_names.append(z_name)
        size_names.append(size_name)

    return pd.DataFrame(data), z_names, size_names


def ratio(value, reference):
    return value / reference if reference > 0 and value > 0 else np.nan


def ppi_efficiency_for_target(runner, target_raw, pi_raw, var_srs_target):
    order = np.argsort(target_raw / pi_raw)
    target_sorted = target_raw[order]
    pi_sorted = pi_raw[order]
    target_over_pi = target_sorted / pi_sorted

    base = runner.Ppi(pi_sorted)
    kernel = base @ base.T
    diag_kernel = np.real(np.diag(kernel))
    variance_matrix = -np.abs(kernel) ** 2
    np.fill_diagonal(variance_matrix, diag_kernel * (1.0 - diag_kernel))
    variance = float(np.real(target_over_pi @ (variance_matrix @ target_over_pi)))
    return var_srs_target / variance if variance > 0 else np.inf


def ppi_efficiency_for_target_in_order(runner, target_raw, pi_raw, var_srs_target, order):
    target_sorted = target_raw[order]
    pi_sorted = pi_raw[order]
    target_over_pi = target_sorted / pi_sorted

    base = runner.Ppi(pi_sorted)
    kernel = base @ base.T
    diag_kernel = np.real(np.diag(kernel))
    variance_matrix = -np.abs(kernel) ** 2
    np.fill_diagonal(variance_matrix, diag_kernel * (1.0 - diag_kernel))
    variance = float(np.real(target_over_pi @ (variance_matrix @ target_over_pi)))
    return var_srs_target / variance if variance > 0 else np.inf


def valid_percent(valid_count, eval_count):
    return 100.0 * valid_count / eval_count if eval_count else 0.0


def run_case(args):
    z_var, size_var, iterations, colony_size, objective, checkpoints = args
    runner = load_runner()
    df_pop, z_names, size_names = make_simulated_population()

    y_raw = df_pop["y"].to_numpy(dtype=float)
    z_raw = df_pop[z_var].to_numpy(dtype=float)
    pi_raw = build_pi_from_size_variable(
        runner,
        df_pop[size_var].to_numpy(dtype=float),
        size_name=size_var,
    )

    sort_idx = np.argsort(z_raw / pi_raw)
    y_sorted = y_raw[sort_idx]
    z_sorted = z_raw[sort_idx]
    pi_sorted = pi_raw[sort_idx]

    n_units = len(df_pop)
    var_srs_y = n_units**2 * (1.0 - N_SAMPLE / n_units) * np.var(y_raw, ddof=1) / N_SAMPLE
    var_srs_z = n_units**2 * (1.0 - N_SAMPLE / n_units) * np.var(z_raw, ddof=1) / N_SAMPLE
    ppi_y_opt_eff = ppi_efficiency_for_target(runner, y_raw, pi_raw, var_srs_y)
    ppi_z_opt_eff = ppi_efficiency_for_target(runner, z_raw, pi_raw, var_srs_z)

    seed = BASE_SEED + 1000 * (z_names.index(z_var) + 1) + (size_names.index(size_var) + 1)

    def make_algorithm(cls, case_suffix, random_state):
        return cls(
            y_sorted=y_sorted,
            z_sorted=z_sorted,
            pik_sorted=pi_sorted,
            var_srs_y=var_srs_y,
            var_srs_z=var_srs_z,
            M=N_SAMPLE,
            n=N_SAMPLE,
            case_name=f"z_{z_var}_pi_from_{size_var}{case_suffix}",
            objective=objective,
            enforce_cadsd_order=True,
            random_state=random_state,
            validation_mode="fast",
            initial_omega_value=INITIAL_OMEGA_VALUE,
            initial_rho_value=INITIAL_RHO_VALUE,
        )

    abc = make_algorithm(runner.ABCAlgorithm, "", seed)
    random_search = make_algorithm(runner.RandomSearchAlgorithm, "_random", seed + 500_000)

    start_evaluator = make_algorithm(runner.ABCAlgorithm, "_start", seed + 900_000)
    start_omega = INITIAL_OMEGA_VALUE * np.ones((N_SAMPLE, n_units))
    start_rho = INITIAL_RHO_VALUE * np.ones((N_SAMPLE, n_units - 1))
    start_food = start_evaluator._food_from_arrays(start_omega, start_rho)
    if start_food is None:
        start_eff_z = np.nan
        start_eff_y = np.nan
    else:
        start_eff_z = start_food["eff_z"]
        start_eff_y = start_food["eff_y"]

    random_start_evals = colony_size
    random_evals_per_iteration = int(round(colony_size * (1.0 + ONLOOKER_FACTOR)))
    corr_y_pi = np.nan if np.std(pi_raw) == 0 else np.corrcoef(y_raw, pi_raw)[0, 1]
    y_over_pi_raw = y_raw / pi_raw
    z_over_pi_raw = z_raw / pi_raw
    corr_y_over_pi_z_over_pi = np.corrcoef(y_over_pi_raw, z_over_pi_raw)[0, 1]

    with contextlib.redirect_stdout(io.StringIO()):
        result = abc.optimize(
            colony_size=colony_size,
            max_iterations=iterations,
            limit=LIMIT,
            verbose=False,
            progress_interval=max(1, iterations // 5),
            local_search_interval=4,
            local_search_attempts=1,
            onlooker_factor=ONLOOKER_FACTOR,
            early_stopping=False,
            min_iterations=1,
            random_searcher=random_search,
            random_start_evals=random_start_evals,
            random_evals_per_iteration=random_evals_per_iteration,
        )

    base_record = {
        "z_variable": z_var,
        "size_variable": size_var,
        "pi_mode": PI_MODE,
        "pi_size_strength": PI_SIZE_STRENGTH,
        "objective": objective,
        "iterations": iterations,
        "colony_size": colony_size,
        "corr_y_z": np.corrcoef(y_raw, z_raw)[0, 1],
        "corr_y_size": np.corrcoef(y_raw, df_pop[size_var].to_numpy(dtype=float))[0, 1],
        "corr_y_pi": corr_y_pi,
        "corr_y_over_pi_z_over_pi": corr_y_over_pi_z_over_pi,
        "Ppi_z_eff": result["optimal_eff_z"],
        "Ppi_y_eff": result["optimal_eff_y"],
        "Ppi_z_opt_eff": ppi_z_opt_eff,
        "Ppi_y_z_order_eff": result["optimal_eff_y"],
        "Ppi_y_opt_eff": ppi_y_opt_eff,
        "Start_z_eff": start_eff_z,
        "Start_y_eff": start_eff_y,
        "Start_z_over_Ppi_z": ratio(start_eff_z, result["optimal_eff_z"]),
        "Start_y_over_Ppi_y": ratio(start_eff_y, result["optimal_eff_y"]),
        "Start_z_over_Ppi_z_opt": ratio(start_eff_z, ppi_z_opt_eff),
        "Start_y_over_Ppi_y_opt": ratio(start_eff_y, ppi_y_opt_eff),
    }

    history_by_iteration = {
        row["iteration"]: row for row in result.get("history_records", [])
    }
    records = []
    for checkpoint in checkpoints:
        row = history_by_iteration.get(checkpoint)
        if row is None:
            available = [i for i in history_by_iteration if i <= checkpoint]
            row = history_by_iteration[max(available)] if available else {}

        record = dict(base_record)
        record["iterations"] = checkpoint
        record.update(
            {
                "ABC_z_eff": row.get("best_eff_z", result["best_eff_z"]),
                "ABC_y_eff": row.get("best_eff_y", result["best_eff_y"]),
                "Random_z_eff": row.get("random_best_eff_z", result["random_best_eff_z"]),
                "Random_y_eff": row.get("random_best_eff_y", result["random_best_eff_y"]),
                "evaluations_ABC": row.get("eval_count", result["eval_count"]),
                "evaluations_Random": row.get("random_eval_count", result["random_eval_count"]),
                "ABC_valid_percent": valid_percent(
                    row.get("valid_count", result["valid_count"]),
                    row.get("eval_count", result["eval_count"]),
                ),
                "Random_valid_percent": valid_percent(
                    row.get("random_valid_count", result["random_valid_count"]),
                    row.get("random_eval_count", result["random_eval_count"]),
                ),
            }
        )
        record["ABC_z_over_Ppi_z"] = ratio(record["ABC_z_eff"], record["Ppi_z_eff"])
        record["ABC_y_over_Ppi_y"] = ratio(record["ABC_y_eff"], record["Ppi_y_eff"])
        record["ABC_z_over_Ppi_z_opt"] = ratio(record["ABC_z_eff"], record["Ppi_z_opt_eff"])
        record["ABC_y_over_Ppi_y_opt"] = ratio(record["ABC_y_eff"], record["Ppi_y_opt_eff"])
        record["Random_z_over_Ppi_z"] = ratio(record["Random_z_eff"], record["Ppi_z_eff"])
        record["Random_y_over_Ppi_y"] = ratio(record["Random_y_eff"], record["Ppi_y_eff"])
        record["Random_z_over_Ppi_z_opt"] = ratio(record["Random_z_eff"], record["Ppi_z_opt_eff"])
        record["Random_y_over_Ppi_y_opt"] = ratio(record["Random_y_eff"], record["Ppi_y_opt_eff"])
        record["both_ABC"] = (
            record["ABC_z_over_Ppi_z_opt"] > 1.0001
            and record["ABC_y_over_Ppi_y_opt"] > 1.0001
        )
        record["both_Random"] = (
            record["Random_z_over_Ppi_z_opt"] > 1.0001
            and record["Random_y_over_Ppi_y_opt"] > 1.0001
        )
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--colony-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--objective", default=OBJECTIVE)
    parser.add_argument("--z-vars", default="all", help="Comma-separated z variables, e.g. z_90,z_80")
    parser.add_argument("--size-vars", default="all", help="Comma-separated size variables, e.g. size_80")
    args = parser.parse_args()

    _, z_names, size_names = make_simulated_population()

    def choose_vars(value, available, label):
        if value.strip().lower() == "all":
            return list(available)
        chosen = [item.strip() for item in value.split(",") if item.strip()]
        bad = [item for item in chosen if item not in available]
        if bad:
            raise ValueError(f"Unknown {label}: {bad}. Available: {available}")
        return chosen

    selected_z_names = choose_vars(args.z_vars, z_names, "z variables")
    selected_size_names = choose_vars(args.size_vars, size_names, "size variables")
    total_cases = len(selected_z_names) * len(selected_size_names)
    checkpoints = list(range(args.checkpoint_interval, args.iterations + 1, args.checkpoint_interval))
    if not checkpoints or checkpoints[-1] != args.iterations:
        checkpoints.append(args.iterations)

    out_dir = PROJECT_ROOT / "simulations_abc" / "jupyters" / "artifacts"
    out_dir.mkdir(exist_ok=True)
    z_tag = "allz" if args.z_vars.strip().lower() == "all" else args.z_vars.replace(",", "-")
    size_tag = "allsizes" if args.size_vars.strip().lower() == "all" else args.size_vars.replace(",", "-")
    run_tag = f"{z_tag}_{size_tag}_{args.objective}_{args.iterations}iter"
    live_path = out_dir / f"abc_random_simulated_live_{run_tag}.csv"

    print("Parallel simulated 9-case checkpoint run")
    print(f"workers       = {args.workers}")
    print(f"checkpoints   = {checkpoints}")
    print(f"colony_size   = {args.colony_size}")
    print(f"objective     = {args.objective}")
    print(f"base_seed     = {BASE_SEED}")
    print(f"z variables   = {selected_z_names}")
    print(f"size variables= {selected_size_names}")
    print(f"live CSV      = {live_path}")
    print()

    start = time.time()
    all_records = []
    final_results = pd.DataFrame()

    cases = [
        (z_var, size_var, args.iterations, args.colony_size, args.objective, checkpoints)
        for z_var in selected_z_names
        for size_var in selected_size_names
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_case = {executor.submit(run_case, case): case for case in cases}
        for i, future in enumerate(as_completed(future_to_case), start=1):
            case_records = future.result()
            all_records.extend(case_records)

            # Save after every completed case. If the run is interrupted,
            # completed case/checkpoint rows are already on disk.
            live_results = (
                pd.DataFrame(all_records)
                .sort_values(["iterations", "z_variable", "size_variable"])
                .reset_index(drop=True)
            )
            live_results.to_csv(live_path, index=False)

            final_record = case_records[-1]
            print(
                f"{i}/{total_cases} done {final_record['z_variable']} / {final_record['size_variable']}: "
                f"Start z*={final_record['Start_z_over_Ppi_z_opt']:.6f}, "
                f"Start y*={final_record['Start_y_over_Ppi_y_opt']:.6f}, "
                f"ABC z*={final_record['ABC_z_over_Ppi_z_opt']:.6f}, "
                f"ABC y*={final_record['ABC_y_over_Ppi_y_opt']:.6f}, "
                f"Rand z*={final_record['Random_z_over_Ppi_z_opt']:.6f}, "
                f"Rand y*={final_record['Random_y_over_Ppi_y_opt']:.6f}, "
                f"both={final_record['both_ABC']} | saved",
                flush=True,
            )

            for checkpoint in checkpoints:
                checkpoint_results = live_results.loc[live_results["iterations"] == checkpoint]
                checkpoint_path = out_dir / f"abc_random_simulated_{z_tag}_{size_tag}_{args.objective}_{checkpoint}iter.csv"
                checkpoint_results.to_csv(checkpoint_path, index=False)

    all_results = (
        pd.DataFrame(all_records)
        .sort_values(["iterations", "z_variable", "size_variable"])
        .reset_index(drop=True)
    )
    for checkpoint in checkpoints:
        checkpoint_results = all_results.loc[all_results["iterations"] == checkpoint]
        checkpoint_path = out_dir / f"abc_random_simulated_{z_tag}_{size_tag}_{args.objective}_{checkpoint}iter.csv"
        checkpoint_results.to_csv(checkpoint_path, index=False)
        print(f"Saved checkpoint: {checkpoint_path}", flush=True)

    final_results = (
        all_results.loc[all_results["iterations"] == args.iterations]
        .sort_values(["z_variable", "size_variable"])
        .reset_index(drop=True)
    )
    print(flush=True)

    cols = [
        "z_variable",
        "size_variable",
        "objective",
        "corr_y_z",
        "corr_y_size",
        "corr_y_pi",
        "corr_y_over_pi_z_over_pi",
        "Ppi_z_eff",
        "Ppi_y_eff",
        "Ppi_z_opt_eff",
        "Ppi_y_z_order_eff",
        "Ppi_y_opt_eff",
        "Start_z_eff",
        "Start_y_eff",
        "ABC_z_eff",
        "ABC_y_eff",
        "Random_z_eff",
        "Random_y_eff",
        "Start_z_over_Ppi_z",
        "Start_y_over_Ppi_y",
        "Start_z_over_Ppi_z_opt",
        "Start_y_over_Ppi_y_opt",
        "ABC_z_over_Ppi_z",
        "ABC_y_over_Ppi_y",
        "ABC_z_over_Ppi_z_opt",
        "ABC_y_over_Ppi_y_opt",
        "Random_z_over_Ppi_z",
        "Random_y_over_Ppi_y",
        "Random_z_over_Ppi_z_opt",
        "Random_y_over_Ppi_y_opt",
        "both_ABC",
        "both_Random",
        "evaluations_ABC",
        "evaluations_Random",
    ]
    print()
    print(final_results[cols].round(6).to_string(index=False))
    print()
    print("ABC both improved cases:", int(final_results["both_ABC"].sum()))
    print("Random both improved cases:", int(final_results["both_Random"].sum()))
    print("Live saved:", live_path)
    print("Total elapsed:", round(time.time() - start, 1), "seconds")


if __name__ == "__main__":
    main()
