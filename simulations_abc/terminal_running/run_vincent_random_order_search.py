"""Diagnostic: optimize a random Vincent ordering.

Here the design is still Vincent/Ppi, but the unit order is allowed to change.
The true Vincent-z reference is the z/pi order. We start from a random order and
let ABC-like mutations search over orderings. Random search receives the same
number of evaluated orderings.
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


def mutate_order(rng, order, target_order=None, guided_prob=0.15):
    order = np.array(order, dtype=int, copy=True)
    n = len(order)

    if target_order is not None and rng.random() < guided_prob:
        target_rank = np.empty(n, dtype=int)
        target_rank[np.asarray(target_order, dtype=int)] = np.arange(n)
        i, j = sorted(rng.choice(n, size=2, replace=False))
        block = order[i : j + 1]
        order[i : j + 1] = block[np.argsort(target_rank[block])]
        return order

    move = rng.integers(0, 3)
    if move == 0:
        i, j = rng.choice(n, size=2, replace=False)
        order[i], order[j] = order[j], order[i]
    elif move == 1:
        i, j = sorted(rng.choice(n, size=2, replace=False))
        order[i : j + 1] = order[i : j + 1][::-1]
    else:
        i, j = rng.choice(n, size=2, replace=False)
        value = order[i]
        order = np.delete(order, i)
        order = np.insert(order, j, value)
    return order


class OrderSearch:
    def __init__(self, vd, runner, y, z, pi, var_srs_y, var_srs_z):
        self.vd = vd
        self.runner = runner
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.var_srs_y = float(var_srs_y)
        self.var_srs_z = float(var_srs_z)

    def evaluate(self, order):
        order = np.asarray(order, dtype=int)
        K = self.vd.vincent_kernel(self.runner, self.pi[order])
        var_z = self.vd.ht_variance(K, self.pi[order], self.z[order])
        var_y = self.vd.ht_variance(K, self.pi[order], self.y[order])
        eff_z = self.vd.efficiency(self.var_srs_z, var_z)
        eff_y = self.vd.efficiency(self.var_srs_y, var_y)
        return {
            "order": order.copy(),
            "var_z": var_z,
            "var_y": var_y,
            "eff_z": eff_z,
            "eff_y": eff_y,
            "score": eff_z,
            "trial": 0,
        }


def run_case(args):
    z_var, pi_source, iterations, checkpoint_interval, colony_size, limit, onlooker_factor, seed = args
    vd = load_base()
    runner = vd.load_runner()
    df_pop, _z_names, _pi_sources = vd.make_simulated_population()

    y = df_pop["y"].to_numpy(dtype=float)
    z = df_pop[z_var].to_numpy(dtype=float)
    pi = vd.build_pi(runner, df_pop, pi_source)
    var_srs_y = vd.srs_variance(y)
    var_srs_z = vd.srs_variance(z)
    search = OrderSearch(vd, runner, y, z, pi, var_srs_y, var_srs_z)

    order_z_opt = np.argsort(z / pi)
    order_y_opt = np.argsort(y / pi)
    rng = np.random.default_rng(seed)
    random_rng = np.random.default_rng(seed + 999)
    start_order = rng.permutation(len(y))

    start = search.evaluate(start_order)
    z_opt = search.evaluate(order_z_opt)
    y_opt = search.evaluate(order_y_opt)

    best = start.copy()
    population = [start]
    evaluations_abc = 1
    while len(population) < colony_size:
        cand_order = mutate_order(rng, start_order, target_order=order_z_opt, guided_prob=0.30)
        cand = search.evaluate(cand_order)
        evaluations_abc += 1
        population.append(cand)
        if cand["score"] > best["score"]:
            best = cand.copy()

    random_best = start.copy()
    evaluations_random = 1
    while evaluations_random < evaluations_abc:
        cand = search.evaluate(random_rng.permutation(len(y)))
        evaluations_random += 1
        if cand["score"] > random_best["score"]:
            random_best = cand.copy()

    checkpoints = []
    for iteration in range(1, iterations + 1):
        before = evaluations_abc
        progress = iteration / max(1, iterations)
        guided_prob = 0.10 + 0.35 * progress

        for i, food in enumerate(list(population)):
            cand = search.evaluate(mutate_order(rng, food["order"], target_order=order_z_opt, guided_prob=guided_prob))
            evaluations_abc += 1
            if cand["score"] > food["score"]:
                population[i] = cand
                if cand["score"] > best["score"]:
                    best = cand.copy()
            else:
                population[i]["trial"] += 1

        scores = np.array([max(0.0, f["score"]) for f in population])
        probabilities = scores / scores.sum() if scores.sum() > 0 else np.full(len(scores), 1 / len(scores))
        for _ in range(int(round(colony_size * onlooker_factor))):
            i = int(rng.choice(len(population), p=probabilities))
            cand = search.evaluate(mutate_order(rng, population[i]["order"], target_order=order_z_opt, guided_prob=guided_prob))
            evaluations_abc += 1
            if cand["score"] > population[i]["score"]:
                population[i] = cand
                if cand["score"] > best["score"]:
                    best = cand.copy()
            else:
                population[i]["trial"] += 1

        for i, food in enumerate(list(population)):
            if food["trial"] >= limit:
                base = best["order"] if rng.random() < 0.7 else start_order
                cand = search.evaluate(mutate_order(rng, base, target_order=order_z_opt, guided_prob=0.55))
                evaluations_abc += 1
                population[i] = cand
                if cand["score"] > best["score"]:
                    best = cand.copy()

        new_abc_evals = evaluations_abc - before
        for _ in range(new_abc_evals):
            cand = search.evaluate(random_rng.permutation(len(y)))
            evaluations_random += 1
            if cand["score"] > random_best["score"]:
                random_best = cand.copy()

        if iteration % checkpoint_interval == 0 or iteration == iterations:
            checkpoints.append(
                {
                    "iteration": iteration,
                    "z_variable": z_var,
                    "pi_source": pi_source,
                    "ABC_z_over_random_start": ratio(best["eff_z"], start["eff_z"]),
                    "ABC_z_over_Vincent_z": ratio(best["eff_z"], z_opt["eff_z"]),
                    "ABC_y_over_Vincent_y": ratio(best["eff_y"], y_opt["eff_y"]),
                    "Random_z_over_random_start": ratio(random_best["eff_z"], start["eff_z"]),
                    "Random_z_over_Vincent_z": ratio(random_best["eff_z"], z_opt["eff_z"]),
                    "Random_y_over_Vincent_y": ratio(random_best["eff_y"], y_opt["eff_y"]),
                    "evaluations_ABC": evaluations_abc,
                    "evaluations_Random": evaluations_random,
                }
            )

    final = checkpoints[-1].copy()
    final.update(
        {
            "corr_y_z": float(np.corrcoef(y, z)[0, 1]),
            "corr_y_pi": 0.0 if pi_source == "equal" else float(np.corrcoef(y, pi)[0, 1]),
            "corr_y_over_pi_z_over_pi": float(np.corrcoef(y / pi, z / pi)[0, 1]),
            "Start_z_over_random_start": 1.0,
            "Start_z_over_Vincent_z": ratio(start["eff_z"], z_opt["eff_z"]),
            "Start_y_over_Vincent_y": ratio(start["eff_y"], y_opt["eff_y"]),
            "Var_z_random_start": start["var_z"],
            "Var_z_Vincent_z": z_opt["var_z"],
            "Var_z_ABC": best["var_z"],
            "Var_z_Random": random_best["var_z"],
        }
    )
    return final, checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--colony-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--onlooker-factor", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    vd = load_base()
    _df, z_names, pi_sources = vd.make_simulated_population()
    cases = []
    for z_i, z_var in enumerate(z_names):
        for p_i, pi_source in enumerate(pi_sources):
            cases.append(
                (
                    z_var,
                    pi_source,
                    args.iterations,
                    args.checkpoint_interval,
                    args.colony_size,
                    args.limit,
                    args.onlooker_factor,
                    args.seed + 1000 * z_i + 17 * p_i,
                )
            )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"vincent_random_order_search_9cases_{args.iterations}iter"
    final_path = ARTIFACT_DIR / f"{stem}.csv"
    checkpoint_path = ARTIFACT_DIR / f"{stem}_checkpoints.csv"

    print("Random Vincent-order search diagnostic")
    print(f"cases              = {len(cases)}")
    print(f"iterations         = {args.iterations}")
    print(f"workers            = {args.workers}")
    print(f"final CSV          = {final_path}")

    start_time = time.time()
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
    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(["iteration", "z_variable", "pi_source"]).reset_index(drop=True)
    final_df.to_csv(final_path, index=False)
    checkpoint_df.to_csv(checkpoint_path, index=False)

    cols = [
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
    print(final_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved final CSV: {final_path}")
    print(f"Saved checkpoints: {checkpoint_path}")
    print(f"Elapsed: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
