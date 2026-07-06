"""Resumable complex CaDSD ABC experiment.

This is the real omega/rho search:
- Vincent/Ppi gives the real-valued benchmark.
- ABC mutates omega and rho.
- CaDsd(omega, rho, pi) builds a possibly complex kernel.
- z is the optimization objective; y is reported as an observed consequence.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import pickle
import sys
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
BASE_RUNNER = PROJECT_ROOT / "simulations_abc" / "terminal_running" / "run_abc_mu284_terminal.py"
ARTIFACT_DIR = PROJECT_ROOT / "simulations_abc" / "jupyters" / "artifacts"
STATE_ROOT = ARTIFACT_DIR / "complex_cadsd_states"

N_UNITS = 50
N_SAMPLE = 5
Z_CORRELATIONS = [0.00, 0.80, 0.90, 0.99]
PI_CORRELATIONS = [0.80, 0.90]
PI_SIZE_STRENGTH = 0.35
BASE_SEED = 202600384


def load_runner():
    spec = importlib.util.spec_from_file_location("cadsd_base_runner", BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def standardize(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    if sd == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def correlated_with_y(y_std, corr, rng):
    noise = rng.normal(size=len(y_std))
    noise = noise - noise.mean()
    noise = noise - np.dot(noise, y_std) / np.dot(y_std, y_std) * y_std
    noise = standardize(noise)
    return corr * y_std + np.sqrt(max(0.0, 1.0 - corr**2)) * noise


def corr_suffix(corr):
    return f"{int(round(corr * 100)):02d}"


def make_simulated_population(seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    location = rng.uniform(80.0, 120.0)
    scale = rng.uniform(8.0, 18.0)
    y = location + scale * rng.normal(size=N_UNITS)
    y_std = standardize(y)

    data = {"unit": np.arange(N_UNITS), "y": y}
    z_names = []
    pi_sources = ["equal"]
    for corr in Z_CORRELATIONS:
        suffix = corr_suffix(corr)
        z_name = f"z_{suffix}"
        data[z_name] = correlated_with_y(y_std, corr, rng)
        z_names.append(z_name)

    for corr in PI_CORRELATIONS:
        suffix = corr_suffix(corr)
        size_name = f"size_{suffix}"
        data[size_name] = correlated_with_y(y_std, corr, rng)
        pi_sources.append(size_name)

    return pd.DataFrame(data), z_names, pi_sources


def make_size_for_pi(x):
    return np.exp(PI_SIZE_STRENGTH * standardize(x))


def build_pi(runner, df_pop, pi_source):
    if pi_source == "equal":
        return np.full(len(df_pop), N_SAMPLE / len(df_pop))
    size = make_size_for_pi(df_pop[pi_source].to_numpy(dtype=float))
    return runner.inclusionprobabilities(size, N_SAMPLE)


def srs_variance(values):
    values = np.asarray(values, dtype=float)
    n_units = len(values)
    return n_units**2 * (1.0 - N_SAMPLE / n_units) * np.var(values, ddof=1) / N_SAMPLE


def ht_variance_from_kernel(K, pi_ordered, values_ordered):
    K = np.asarray(K, dtype=np.complex128)
    pi_ordered = np.asarray(pi_ordered, dtype=float)
    values_ordered = np.asarray(values_ordered, dtype=float)
    diag_K = np.real(np.diag(K))
    A = -np.abs(K) ** 2
    np.fill_diagonal(A, diag_K * (1.0 - diag_K))
    values_over_pi = values_ordered / pi_ordered
    return float(np.real(values_over_pi @ (A @ values_over_pi)))


def ppi_variance_for_order(runner, values, pi, order):
    K = runner.Ppi(pi[order]) @ runner.Ppi(pi[order]).T
    return ht_variance_from_kernel(K, pi[order], values[order])


def cadsd_variance_pair(runner, y_desc, z_desc, pi_desc, omega, rho):
    K = runner.CaDsd(pi=pi_desc, M=N_SAMPLE, omega=omega, rho=rho)["K"]
    return (
        ht_variance_from_kernel(K, pi_desc, y_desc),
        ht_variance_from_kernel(K, pi_desc, z_desc),
    )


def ratio(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return float(value / reference)


def case_state_path(stem, z_var, pi_source):
    return STATE_ROOT / stem / f"{z_var}__{pi_source}.pkl"


def checkpoint_row(
    iteration,
    z_var,
    pi_source,
    corr_y_z,
    corr_y_pi,
    corr_ratio,
    start_eff_z,
    start_eff_y,
    vincent_eff_z,
    vincent_y_eff_y,
    abc,
    random_searcher,
):
    return {
        "iteration": iteration,
        "z_variable": z_var,
        "pi_source": pi_source,
        "corr_y_z": corr_y_z,
        "corr_y_pi": corr_y_pi,
        "corr_y_over_pi_z_over_pi": corr_ratio,
        "Start_z_over_Vincent_z": ratio(start_eff_z, vincent_eff_z),
        "Start_y_over_Vincent_y": ratio(start_eff_y, vincent_y_eff_y),
        "ABC_z_over_Vincent_z": ratio(abc.global_best_eff_z, vincent_eff_z),
        "ABC_y_over_Vincent_y": ratio(abc.global_best_eff_y, vincent_y_eff_y),
        "Random_z_over_Vincent_z": ratio(random_searcher.global_best_eff_z, vincent_eff_z),
        "Random_y_over_Vincent_y": ratio(random_searcher.global_best_eff_y, vincent_y_eff_y),
        "ABC_eff_z": abc.global_best_eff_z,
        "ABC_eff_y": abc.global_best_eff_y,
        "Random_eff_z": random_searcher.global_best_eff_z,
        "Random_eff_y": random_searcher.global_best_eff_y,
        "Vincent_eff_z": vincent_eff_z,
        "Vincent_y_eff_y": vincent_y_eff_y,
        "evaluations_ABC": abc.eval_count,
        "evaluations_Random": random_searcher.eval_count,
        "valid_ABC": abc.valid_count,
        "valid_Random": random_searcher.valid_count,
    }


def run_case(args):
    (
        stem,
        z_var,
        pi_source,
        target_iterations,
        checkpoint_interval,
        colony_size,
        limit,
        onlooker_factor,
        local_search_interval,
        local_search_attempts,
        validation_mode,
        seed,
        resume,
    ) = args

    runner = load_runner()
    df_pop, _z_names, _pi_sources = make_simulated_population()
    y_raw = df_pop["y"].to_numpy(dtype=float)
    z_raw = df_pop[z_var].to_numpy(dtype=float)
    pi_raw = build_pi(runner, df_pop, pi_source)

    order_desc_pi = np.argsort(pi_raw)[::-1]
    order_y = np.argsort(y_raw / pi_raw)

    y_desc = y_raw[order_desc_pi]
    z_desc = z_raw[order_desc_pi]
    pi_desc = pi_raw[order_desc_pi]

    var_srs_y = srs_variance(y_raw)
    var_srs_z = srs_variance(z_raw)

    vincent_var_z = ppi_variance_for_order(runner, z_raw, pi_raw, order_desc_pi)
    vincent_eff_z = var_srs_z / vincent_var_z
    vincent_y_var_y = ppi_variance_for_order(runner, y_raw, pi_raw, order_y)
    vincent_y_eff_y = var_srs_y / vincent_y_var_y

    center_omega = np.zeros((N_SAMPLE, len(pi_desc)))
    center_rho = 0.5 * np.ones((N_SAMPLE, len(pi_desc) - 1))
    start_var_y, start_var_z = cadsd_variance_pair(
        runner, y_desc, z_desc, pi_desc, center_omega, center_rho
    )
    start_eff_y = var_srs_y / start_var_y
    start_eff_z = var_srs_z / start_var_z

    corr_y_z = float(np.corrcoef(y_raw, z_raw)[0, 1])
    corr_y_pi = 0.0 if pi_source == "equal" else float(np.corrcoef(y_raw, pi_raw)[0, 1])
    corr_ratio = float(np.corrcoef(y_raw / pi_raw, z_raw / pi_raw)[0, 1])

    path = case_state_path(stem, z_var, pi_source)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and path.exists():
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        abc = saved["abc"]
        random_searcher = saved["random_searcher"]
        population = saved["population"]
        current_iteration = int(saved["iteration"])
        total_abandoned = int(saved.get("total_abandoned", 0))
    else:
        abc = runner.ABCAlgorithm(
            y_desc,
            z_desc,
            pi_desc,
            var_srs_y,
            var_srs_z,
            M=N_SAMPLE,
            n=N_SAMPLE,
            case_name=f"{z_var}_{pi_source}",
            objective="eff_z",
            enforce_cadsd_order=False,
            random_state=seed,
            validation_mode=validation_mode,
            initial_omega_value=0.0,
            initial_rho_value=0.5,
        )
        random_searcher = runner.RandomSearchAlgorithm(
            y_desc,
            z_desc,
            pi_desc,
            var_srs_y,
            var_srs_z,
            M=N_SAMPLE,
            n=N_SAMPLE,
            case_name=f"random_{z_var}_{pi_source}",
            objective="eff_z",
            enforce_cadsd_order=False,
            random_state=seed + 999,
            validation_mode=validation_mode,
            initial_omega_value=0.0,
            initial_rho_value=0.5,
        )
        population = abc.initialize_population(colony_size, verbose=False)
        random_searcher.step(abc.eval_count, include_center_once=True)
        current_iteration = 0
        total_abandoned = 0

    checkpoint_rows = []
    if current_iteration >= target_iterations:
        row = checkpoint_row(
            current_iteration,
            z_var,
            pi_source,
            corr_y_z,
            corr_y_pi,
            corr_ratio,
            start_eff_z,
            start_eff_y,
            vincent_eff_z,
            vincent_y_eff_y,
            abc,
            random_searcher,
        )
        return row, [row], current_iteration

    start_time = time.time()
    for iteration in range(current_iteration + 1, target_iterations + 1):
        progress = iteration / max(1, target_iterations)
        before_eval_count = abc.eval_count

        population = abc.employed_bee_phase(population, progress)
        population = abc.onlooker_bee_phase(population, progress, onlooker_factor=onlooker_factor)
        population, n_abandoned = abc.scout_bee_phase(population, limit, progress)
        total_abandoned += n_abandoned

        if local_search_interval and iteration % local_search_interval == 0:
            population = abc.local_search_phase(
                population, progress, attempts=local_search_attempts
            )

        population = abc._inject_elite(population)

        new_abc_evals = abc.eval_count - before_eval_count
        random_searcher.step(new_abc_evals, include_center_once=False)

        record = {
            "iteration": iteration,
            "best_eff_z": abc.global_best_eff_z,
            "best_eff_y": abc.global_best_eff_y,
            "best_score": abc.global_best_score,
            "n_abandoned": n_abandoned,
            "eval_count": abc.eval_count,
            "valid_count": abc.valid_count,
            "random_best_eff_z": random_searcher.global_best_eff_z,
            "random_best_eff_y": random_searcher.global_best_eff_y,
            "random_eval_count": random_searcher.eval_count,
            "random_valid_count": random_searcher.valid_count,
            "elapsed": time.time() - start_time,
        }
        abc.history_records.append(record)

        if iteration % checkpoint_interval == 0 or iteration == target_iterations:
            row = checkpoint_row(
                iteration,
                z_var,
                pi_source,
                corr_y_z,
                corr_y_pi,
                corr_ratio,
                start_eff_z,
                start_eff_y,
                vincent_eff_z,
                vincent_y_eff_y,
                abc,
                random_searcher,
            )
            checkpoint_rows.append(row)
            with path.open("wb") as handle:
                pickle.dump(
                    {
                        "iteration": iteration,
                        "abc": abc,
                        "random_searcher": random_searcher,
                        "population": population,
                        "total_abandoned": total_abandoned,
                    },
                    handle,
                )

    return checkpoint_rows[-1], checkpoint_rows, target_iterations


def parse_list(value):
    if value == "all":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--colony-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--onlooker-factor", type=float, default=0.5)
    parser.add_argument("--local-search-interval", type=int, default=10)
    parser.add_argument("--local-search-attempts", type=int, default=1)
    parser.add_argument("--validation-mode", default="fast", choices=["fast", "strict"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--z-vars", default="all")
    parser.add_argument("--pi-sources", default="all")
    args = parser.parse_args()

    _runner = load_runner()
    _df_pop, z_names, pi_sources = make_simulated_population()
    selected_z = parse_list(args.z_vars) or z_names
    selected_pi = parse_list(args.pi_sources) or pi_sources

    stem = (
        f"complex_cadsd_abc_{args.colony_size}col_"
        f"{args.validation_mode}"
    )
    final_path = ARTIFACT_DIR / f"{stem}_{args.iterations}iter.csv"
    checkpoint_path = ARTIFACT_DIR / f"{stem}_{args.iterations}iter_checkpoints.csv"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    cases = []
    for z_i, z_var in enumerate(selected_z):
        for p_i, pi_source in enumerate(selected_pi):
            cases.append(
                (
                    stem,
                    z_var,
                    pi_source,
                    args.iterations,
                    args.checkpoint_interval,
                    args.colony_size,
                    args.limit,
                    args.onlooker_factor,
                    args.local_search_interval,
                    args.local_search_attempts,
                    args.validation_mode,
                    args.seed + 1000 * z_i + 17 * p_i,
                    args.resume,
                )
            )

    print("Complex CaDSD ABC omega/rho experiment")
    print(f"cases              = {len(cases)}")
    print(f"iterations target  = {args.iterations}")
    print(f"checkpoint interval= {args.checkpoint_interval}")
    print(f"colony size        = {args.colony_size}")
    print(f"resume             = {args.resume}")
    print(f"state dir          = {STATE_ROOT / stem}")
    print(f"final CSV          = {final_path}")

    start = time.time()
    finals = []
    checkpoint_rows = []

    if args.workers <= 1:
        for case in cases:
            final, rows, current = run_case(case)
            finals.append(final)
            checkpoint_rows.extend(rows)
            print(
                f"{len(finals)}/{len(cases)} {final['z_variable']} / {final['pi_source']} "
                f"iter={current}: ABC_z/V={final['ABC_z_over_Vincent_z']:.4f}, "
                f"ABC_y/Vy={final['ABC_y_over_Vincent_y']:.4f}, "
                f"Rnd_z/V={final['Random_z_over_Vincent_z']:.4f}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_case, case): case for case in cases}
            for future in as_completed(future_map):
                final, rows, current = future.result()
                finals.append(final)
                checkpoint_rows.extend(rows)
                print(
                    f"{len(finals)}/{len(cases)} {final['z_variable']} / {final['pi_source']} "
                    f"iter={current}: ABC_z/V={final['ABC_z_over_Vincent_z']:.4f}, "
                    f"ABC_y/Vy={final['ABC_y_over_Vincent_y']:.4f}, "
                    f"Rnd_z/V={final['Random_z_over_Vincent_z']:.4f}"
                )

    final_df = pd.DataFrame(finals).sort_values(["z_variable", "pi_source"]).reset_index(drop=True)
    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(
        ["iteration", "z_variable", "pi_source"]
    ).reset_index(drop=True)
    final_df.to_csv(final_path, index=False)
    checkpoint_df.to_csv(checkpoint_path, index=False)

    display_cols = [
        "z_variable",
        "pi_source",
        "corr_y_z",
        "corr_y_pi",
        "corr_y_over_pi_z_over_pi",
        "Start_z_over_Vincent_z",
        "Start_y_over_Vincent_y",
        "ABC_z_over_Vincent_z",
        "ABC_y_over_Vincent_y",
        "Random_z_over_Vincent_z",
        "Random_y_over_Vincent_y",
        "evaluations_ABC",
        "evaluations_Random",
        "valid_ABC",
        "valid_Random",
    ]
    print()
    print(final_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved final CSV: {final_path}")
    print(f"Saved checkpoint CSV: {checkpoint_path}")
    print(f"Elapsed: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
