"""Diagnostic: start from a random Vincent/Ppi order instead of z/pi order.

Purpose:
If ABC improves this random Vincent design, but not the z/pi-ordered Vincent
design, it confirms that the search can improve bad starts and that the
Vincent z-ordered design is already locally/empirically optimal for z.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = PROJECT_ROOT / "simulations_abc" / "terminal_running" / "run_vincent_direct_9cases.py"
ARTIFACT_DIR = PROJECT_ROOT / "simulations_abc" / "jupyters" / "artifacts"


def load_base():
    spec = importlib.util.spec_from_file_location("vincent_direct_base", BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ratio(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return float(value / reference)


def run_case(args):
    z_var, pi_source, iterations, checkpoint_interval, colony_size, limit, onlooker_factor, random_order_seed = args
    vd = load_base()
    runner = vd.load_runner()
    df_pop, _z_names, _size_names = vd.make_simulated_population()

    y_raw = df_pop["y"].to_numpy(dtype=float)
    z_raw = df_pop[z_var].to_numpy(dtype=float)
    pi_raw = vd.build_pi(runner, df_pop, pi_source)

    order_random_rng = np.random.default_rng(random_order_seed)
    order_start = order_random_rng.permutation(len(df_pop))
    order_z_opt = np.argsort(z_raw / pi_raw)
    order_y_opt = np.argsort(y_raw / pi_raw)

    K_start = vd.vincent_kernel(runner, pi_raw[order_start])
    K_z_opt = vd.vincent_kernel(runner, pi_raw[order_z_opt])
    K_y_opt = vd.vincent_kernel(runner, pi_raw[order_y_opt])

    var_srs_y = vd.srs_variance(y_raw)
    var_srs_z = vd.srs_variance(z_raw)

    start_var_z = vd.ht_variance(K_start, pi_raw[order_start], z_raw[order_start])
    start_var_y = vd.ht_variance(K_start, pi_raw[order_start], y_raw[order_start])
    z_opt_var_z = vd.ht_variance(K_z_opt, pi_raw[order_z_opt], z_raw[order_z_opt])
    y_opt_var_y = vd.ht_variance(K_y_opt, pi_raw[order_y_opt], y_raw[order_y_opt])

    start_eff_z = vd.efficiency(var_srs_z, start_var_z)
    start_eff_y = vd.efficiency(var_srs_y, start_var_y)
    z_opt_eff_z = vd.efficiency(var_srs_z, z_opt_var_z)
    y_opt_eff_y = vd.efficiency(var_srs_y, y_opt_var_y)

    abc_seed = random_order_seed + 1
    random_seed = random_order_seed + 2
    abc = vd.DirectABC(
        K_start,
        pi_raw[order_start],
        y_raw[order_start],
        z_raw[order_start],
        var_srs_y,
        var_srs_z,
        seed=abc_seed,
        colony_size=colony_size,
        limit=limit,
        onlooker_factor=onlooker_factor,
    )
    rnd = vd.DirectRandom(
        K_start,
        pi_raw[order_start],
        y_raw[order_start],
        z_raw[order_start],
        var_srs_y,
        var_srs_z,
        seed=random_seed,
    )

    abc.initialize()
    rnd.step(abc.evaluations)

    checkpoint_rows = []
    for iteration in range(1, iterations + 1):
        new_evals = abc.step(iteration, iterations)
        rnd.step(new_evals)

        if iteration % checkpoint_interval == 0 or iteration == iterations:
            checkpoint_rows.append(
                {
                    "iteration": iteration,
                    "z_variable": z_var,
                    "pi_source": pi_source,
                    "ABC_z_over_random_start": ratio(abc.best_eff_z, start_eff_z),
                    "ABC_z_over_Vincent_z": ratio(abc.best_eff_z, z_opt_eff_z),
                    "ABC_y_over_Vincent_y": ratio(abc.best_eff_y, y_opt_eff_y),
                    "Random_z_over_random_start": ratio(rnd.best_eff_z, start_eff_z),
                    "Random_z_over_Vincent_z": ratio(rnd.best_eff_z, z_opt_eff_z),
                    "Random_y_over_Vincent_y": ratio(rnd.best_eff_y, y_opt_eff_y),
                    "evaluations_ABC": abc.evaluations,
                    "evaluations_Random": rnd.evaluations,
                }
            )

    final = checkpoint_rows[-1].copy()
    final.update(
        {
            "corr_y_z": float(np.corrcoef(y_raw, z_raw)[0, 1]),
            "corr_y_pi": 0.0 if pi_source == "equal" else float(np.corrcoef(y_raw, pi_raw)[0, 1]),
            "corr_y_over_pi_z_over_pi": float(np.corrcoef(y_raw / pi_raw, z_raw / pi_raw)[0, 1]),
            "Start_z_over_random_start": 1.0,
            "Start_z_over_Vincent_z": ratio(start_eff_z, z_opt_eff_z),
            "Start_y_over_Vincent_y": ratio(start_eff_y, y_opt_eff_y),
            "Var_z_random_start": start_var_z,
            "Var_z_Vincent_z": z_opt_var_z,
            "Var_z_ABC": abc.best_var_z,
            "Var_z_Random": rnd.best_var_z,
        }
    )
    return final, checkpoint_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--colony-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--onlooker-factor", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-order-seed", type=int, default=20260706)
    args = parser.parse_args()

    vd = load_base()
    _df_pop, z_names, pi_sources = vd.make_simulated_population()
    cases = []
    for z_i, z_var in enumerate(z_names):
        for pi_i, pi_source in enumerate(pi_sources):
            seed = args.random_order_seed + 1000 * z_i + 17 * pi_i
            cases.append(
                (
                    z_var,
                    pi_source,
                    args.iterations,
                    args.checkpoint_interval,
                    args.colony_size,
                    args.limit,
                    args.onlooker_factor,
                    seed,
                )
            )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"vincent_random_start_diagnostic_9cases_{args.iterations}iter"
    final_path = ARTIFACT_DIR / f"{stem}.csv"
    checkpoints_path = ARTIFACT_DIR / f"{stem}_checkpoints.csv"

    print("Random-start Vincent diagnostic")
    print(f"cases              = {len(cases)}")
    print(f"iterations         = {args.iterations}")
    print(f"checkpoint interval= {args.checkpoint_interval}")
    print(f"workers            = {args.workers}")
    print(f"final CSV          = {final_path}")

    start = time.time()
    finals = []
    checkpoint_rows = []
    if args.workers <= 1:
        for case in cases:
            final, checkpoints = run_case(case)
            finals.append(final)
            checkpoint_rows.extend(checkpoints)
            print(
                f"{len(finals)}/{len(cases)} {final['z_variable']} / {final['pi_source']}: "
                f"ABC z/random={final['ABC_z_over_random_start']:.3f}, "
                f"ABC z/V={final['ABC_z_over_Vincent_z']:.3f}, "
                f"Rnd z/random={final['Random_z_over_random_start']:.3f}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_case, case): case for case in cases}
            for future in as_completed(future_map):
                final, checkpoints = future.result()
                finals.append(final)
                checkpoint_rows.extend(checkpoints)
                print(
                    f"{len(finals)}/{len(cases)} {final['z_variable']} / {final['pi_source']}: "
                    f"ABC z/random={final['ABC_z_over_random_start']:.3f}, "
                    f"ABC z/V={final['ABC_z_over_Vincent_z']:.3f}, "
                    f"Rnd z/random={final['Random_z_over_random_start']:.3f}"
                )

    final_df = pd.DataFrame(finals).sort_values(["z_variable", "pi_source"]).reset_index(drop=True)
    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(
        ["iteration", "z_variable", "pi_source"]
    ).reset_index(drop=True)
    final_df.to_csv(final_path, index=False)
    checkpoint_df.to_csv(checkpoints_path, index=False)

    display_cols = [
        "z_variable",
        "pi_source",
        "corr_y_z",
        "corr_y_pi",
        "corr_y_over_pi_z_over_pi",
        "Start_z_over_Vincent_z",
        "Start_y_over_Vincent_y",
        "ABC_z_over_random_start",
        "ABC_z_over_Vincent_z",
        "ABC_y_over_Vincent_y",
        "Random_z_over_random_start",
        "Random_z_over_Vincent_z",
        "Random_y_over_Vincent_y",
        "evaluations_ABC",
        "evaluations_Random",
    ]
    print()
    print(final_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved final CSV: {final_path}")
    print(f"Saved checkpoints: {checkpoints_path}")
    print(f"Elapsed: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
