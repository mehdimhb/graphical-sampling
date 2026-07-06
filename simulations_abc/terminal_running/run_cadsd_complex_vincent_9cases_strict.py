"""Strict 9-case complex CaDSD ABC experiment.

This script is intentionally conservative:
- each case satisfies Vincent's structure for the optimized variable z:
  pi is non-increasing and z/pi is increasing;
- the ABC and random searches both change omega/rho only;
- every candidate kernel is checked with strict eigenvalue validation;
- random search receives the same number of evaluations as ABC;
- y efficiencies are reported for ABC and random as well.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
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

N_UNITS = 50
N_SAMPLE = 7
BASE_SEED = 20260706
RATIO_CORRELATIONS = [0.00, 0.80, 0.90]
PI_TARGETS = ["equal", "pi80", "pi90"]
Y_RATIO_SCALE = 10.0


def load_runner():
    spec = importlib.util.spec_from_file_location("cadsd_base_runner_9strict", BASE_RUNNER)
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


def correlated_with_target(target_std, corr, rng):
    noise = rng.normal(size=len(target_std))
    noise = noise - noise.mean()
    noise = noise - np.dot(noise, target_std) / np.dot(target_std, target_std) * target_std
    noise = standardize(noise)
    return corr * target_std + np.sqrt(max(0.0, 1.0 - corr**2)) * noise


def corr_suffix(corr):
    return f"{int(round(corr * 100)):02d}"


def srs_variance(values):
    values = np.asarray(values, dtype=float)
    n_units = len(values)
    return n_units**2 * (1.0 - N_SAMPLE / n_units) * np.var(values, ddof=1) / N_SAMPLE


def ht_variance(K, pi, values):
    K = np.asarray(K, dtype=np.complex128)
    diag_K = np.real(np.diag(K))
    A = -np.abs(K) ** 2
    np.fill_diagonal(A, diag_K * (1.0 - diag_K))
    values_over_pi = values / pi
    return float(np.real(values_over_pi @ (A @ values_over_pi)))


def ratio(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return float(value / reference)


def ppi_eff_for_y_reference(runner, y, pi, var_srs_y):
    order_y = np.argsort(y / pi)
    K_y = runner.Ppi(pi[order_y]) @ runner.Ppi(pi[order_y]).T
    var_y = ht_variance(K_y, pi[order_y], y[order_y])
    return var_srs_y / var_y


def make_pi(pi_target, rng):
    if pi_target == "equal":
        return np.full(N_UNITS, N_SAMPLE / N_UNITS)

    pi = rng.uniform(0.4, 0.8, size=N_UNITS)
    pi = pi * N_SAMPLE / pi.sum()
    return np.sort(pi)[::-1]


def choose_y_ratio_base(pi, y_ratio_std, pi_target):
    if pi_target == "equal":
        return 100.0

    target = 0.80 if pi_target == "pi80" else 0.90
    candidates = np.geomspace(1.0, 1.0e5, 600)
    best_base = candidates[0]
    best_error = np.inf

    for base in candidates:
        y_ratio = base + Y_RATIO_SCALE * y_ratio_std
        if np.min(y_ratio) <= 0:
            continue
        y = pi * y_ratio
        corr = np.corrcoef(y, pi)[0, 1]
        error = abs(corr - target)
        if error < best_error:
            best_error = error
            best_base = base

    return float(best_base)


def make_case(ratio_corr, pi_target, seed=BASE_SEED):
    seed_shift = 10_000 * list(RATIO_CORRELATIONS).index(ratio_corr) + list(PI_TARGETS).index(pi_target)
    rng = np.random.default_rng(seed + seed_shift)

    pi = make_pi(pi_target, rng)
    first_ratio = 5.0 / pi[0]
    z_ratio = np.cumsum(np.r_[first_ratio, rng.uniform(0.2, 1.0, size=N_UNITS - 1)])
    z = z_ratio * pi

    y_ratio_std = correlated_with_target(standardize(z_ratio), ratio_corr, rng)
    base = choose_y_ratio_base(pi, y_ratio_std, pi_target)
    y_ratio = base + Y_RATIO_SCALE * y_ratio_std
    y = y_ratio * pi

    checks = {
        "pi_nonincreasing": bool(np.all(np.diff(pi) <= 1e-12)),
        "z_over_pi_increasing": bool(np.all(np.diff(z / pi) > 0)),
    }
    return y, z, pi, checks


def checkpoint_row(iteration, ratio_corr, pi_target, checks, abc, rnd, refs, start_refs):
    return {
        "iteration": iteration,
        "z_case": f"zratio_{corr_suffix(ratio_corr)}",
        "pi_case": pi_target,
        "target_corr_y_over_pi_z_over_pi": ratio_corr,
        "pi_nonincreasing": checks["pi_nonincreasing"],
        "z_over_pi_increasing": checks["z_over_pi_increasing"],
        "corr_y_z": refs["corr_y_z"],
        "corr_y_pi": refs["corr_y_pi"],
        "corr_y_over_pi_z_over_pi": refs["corr_ratio"],
        "Start_z_over_Vincent_z": ratio(start_refs["start_eff_z"], refs["vincent_eff_z"]),
        "Start_y_over_Vincent_y": ratio(start_refs["start_eff_y"], refs["vincent_y_eff_y"]),
        "ABC_z_over_Vincent_z": ratio(abc.global_best_eff_z, refs["vincent_eff_z"]),
        "ABC_y_over_Vincent_y": ratio(abc.global_best_eff_y, refs["vincent_y_eff_y"]),
        "Random_z_over_Vincent_z": ratio(rnd.global_best_eff_z, refs["vincent_eff_z"]),
        "Random_y_over_Vincent_y": ratio(rnd.global_best_eff_y, refs["vincent_y_eff_y"]),
        "evaluations_ABC": abc.eval_count,
        "evaluations_Random": rnd.eval_count,
        "valid_ABC": abc.valid_count,
        "valid_Random": rnd.valid_count,
    }


def append_live_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not path.exists())


def run_case(args):
    (
        stem,
        ratio_corr,
        pi_target,
        iterations,
        checkpoint_interval,
        colony_size,
        limit,
        onlooker_factor,
        validation_mode,
        live_path,
        verbose,
    ) = args
    runner = load_runner()
    y, z, pi, checks = make_case(ratio_corr, pi_target)

    if not checks["pi_nonincreasing"] or not checks["z_over_pi_increasing"]:
        raise ValueError(f"Vincent checks failed for {ratio_corr=}, {pi_target=}: {checks}")

    var_srs_y = srs_variance(y)
    var_srs_z = srs_variance(z)

    K_vincent = runner.Ppi(pi) @ runner.Ppi(pi).T
    vincent_var_z = ht_variance(K_vincent, pi, z)
    vincent_eff_z = var_srs_z / vincent_var_z
    vincent_y_eff_y = ppi_eff_for_y_reference(runner, y, pi, var_srs_y)

    center_omega = np.zeros((N_SAMPLE, N_UNITS))
    center_rho = 0.5 * np.ones((N_SAMPLE, N_UNITS - 1))
    center_K = runner.CaDsd(pi=pi, M=N_SAMPLE, omega=center_omega, rho=center_rho)["K"]
    start_eff_z = var_srs_z / ht_variance(center_K, pi, z)
    start_eff_y = var_srs_y / ht_variance(center_K, pi, y)

    refs = {
        "vincent_eff_z": vincent_eff_z,
        "vincent_y_eff_y": vincent_y_eff_y,
        "corr_y_z": float(np.corrcoef(y, z)[0, 1]),
        "corr_y_pi": 0.0 if pi_target == "equal" else float(np.corrcoef(y, pi)[0, 1]),
        "corr_ratio": float(np.corrcoef(y / pi, z / pi)[0, 1]),
    }
    start_refs = {"start_eff_z": start_eff_z, "start_eff_y": start_eff_y}

    seed = BASE_SEED + 1000 * list(RATIO_CORRELATIONS).index(ratio_corr) + 37 * list(PI_TARGETS).index(pi_target)
    case_name = f"{stem}_{corr_suffix(ratio_corr)}_{pi_target}"
    abc = runner.ABCAlgorithm(
        y,
        z,
        pi,
        var_srs_y,
        var_srs_z,
        M=N_SAMPLE,
        n=N_SAMPLE,
        case_name=case_name,
        objective="eff_z",
        enforce_cadsd_order=False,
        random_state=seed,
        validation_mode=validation_mode,
        initial_omega_value=0.0,
        initial_rho_value=0.5,
    )
    rnd = runner.RandomSearchAlgorithm(
        y,
        z,
        pi,
        var_srs_y,
        var_srs_z,
        M=N_SAMPLE,
        n=N_SAMPLE,
        case_name=f"random_{case_name}",
        objective="eff_z",
        enforce_cadsd_order=False,
        random_state=seed + 999,
        validation_mode=validation_mode,
        initial_omega_value=0.0,
        initial_rho_value=0.5,
    )

    population = abc.initialize_population(colony_size, verbose=False)
    rnd.step(abc.eval_count, include_center_once=True)

    checkpoints = []
    for iteration in range(1, iterations + 1):
        progress = iteration / max(1, iterations)
        before = abc.eval_count
        population = abc.employed_bee_phase(population, progress)
        population = abc.onlooker_bee_phase(population, progress, onlooker_factor=onlooker_factor)
        population, _n_abandoned = abc.scout_bee_phase(population, limit, progress)
        if iteration % 10 == 0:
            population = abc.local_search_phase(population, progress, attempts=1)
        population = abc._inject_elite(population)
        rnd.step(abc.eval_count - before, include_center_once=False)

        if iteration % checkpoint_interval == 0 or iteration == iterations:
            row = checkpoint_row(iteration, ratio_corr, pi_target, checks, abc, rnd, refs, start_refs)
            checkpoints.append(row)
            append_live_row(live_path, row)
            if verbose:
                print(
                    f"{row['z_case']} / {pi_target} iter={iteration:4d}: "
                    f"ABC_z/V={row['ABC_z_over_Vincent_z']:.4f}, "
                    f"ABC_y/Vy={row['ABC_y_over_Vincent_y']:.4f}, "
                    f"Rnd_z/V={row['Random_z_over_Vincent_z']:.4f}, "
                    f"Rnd_y/Vy={row['Random_y_over_Vincent_y']:.4f}",
                    flush=True,
                )

    return checkpoints[-1], checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--colony-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--onlooker-factor", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--validation-mode", choices=["fast", "strict"], default="strict")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"complex_cadsd_vincent_9cases_strict_{args.iterations}iter"
    final_path = ARTIFACT_DIR / f"{stem}.csv"
    checkpoint_path = ARTIFACT_DIR / f"{stem}_checkpoints.csv"
    live_path = ARTIFACT_DIR / f"{stem}_live.csv"
    if live_path.exists():
        live_path.unlink()

    cases = [
        (
            stem,
            ratio_corr,
            pi_target,
            args.iterations,
            args.checkpoint_interval,
            args.colony_size,
            args.limit,
            args.onlooker_factor,
            args.validation_mode,
            live_path,
            args.workers <= 1,
        )
        for ratio_corr in RATIO_CORRELATIONS
        for pi_target in PI_TARGETS
    ]

    print("Strict Vincent 9-case complex CaDSD ABC")
    print(f"iterations  = {args.iterations}")
    print(f"cases       = {len(cases)}")
    print(f"validation  = {args.validation_mode}")
    print(f"final CSV   = {final_path}")
    print(f"live CSV    = {live_path}")

    start = time.time()
    finals = []
    checkpoint_rows = []
    if args.workers <= 1:
        for case in cases:
            final, rows = run_case(case)
            finals.append(final)
            checkpoint_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_case, case): case for case in cases}
            for future in as_completed(future_map):
                final, rows = future.result()
                finals.append(final)
                checkpoint_rows.extend(rows)
                print(
                    f"{len(finals)}/{len(cases)} {final['z_case']} / {final['pi_case']}: "
                    f"ABC_z/V={final['ABC_z_over_Vincent_z']:.4f}, "
                    f"ABC_y/Vy={final['ABC_y_over_Vincent_y']:.4f}, "
                    f"Rnd_y/Vy={final['Random_y_over_Vincent_y']:.4f}",
                    flush=True,
                )

    final_df = pd.DataFrame(finals).sort_values(["target_corr_y_over_pi_z_over_pi", "pi_case"])
    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(
        ["iteration", "target_corr_y_over_pi_z_over_pi", "pi_case"]
    )
    final_df.to_csv(final_path, index=False)
    checkpoint_df.to_csv(checkpoint_path, index=False)

    display_cols = [
        "iteration",
        "z_case",
        "pi_case",
        "pi_nonincreasing",
        "z_over_pi_increasing",
        "corr_y_z",
        "corr_y_pi",
        "corr_y_over_pi_z_over_pi",
        "Start_z_over_Vincent_z",
        "Start_y_over_Vincent_y",
        "ABC_z_over_Vincent_z",
        "ABC_y_over_Vincent_y",
        "Random_z_over_Vincent_z",
        "Random_y_over_Vincent_y",
    ]
    print()
    print(final_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"Saved final CSV: {final_path}")
    print(f"Saved checkpoint CSV: {checkpoint_path}")
    print(f"Elapsed: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
