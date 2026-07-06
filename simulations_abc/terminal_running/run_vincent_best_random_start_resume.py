"""Resumable best-random-Vincent-start experiment.

For each scenario:
1. generate many random Vincent/Ppi orderings;
2. pick the best random start by z efficiency;
3. run ABC order search and equal-budget random search from that same start;
4. save checkpoints and per-case state so the run can continue later.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDER_RUNNER = PROJECT_ROOT / "simulations_abc" / "terminal_running" / "run_vincent_random_order_search.py"
ARTIFACT_DIR = PROJECT_ROOT / "simulations_abc" / "jupyters" / "artifacts"
STATE_DIR = ARTIFACT_DIR / "best_random_start_states"


def load_order_runner():
    spec = importlib.util.spec_from_file_location("order_runner", ORDER_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ratio(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return float(value / reference)


def case_key(z_var, pi_source):
    return f"{z_var}__{pi_source}"


def state_path(stem, z_var, pi_source):
    return STATE_DIR / stem / f"{case_key(z_var, pi_source)}.pkl"


def evaluate_best_random_start(search, rng, n_starts):
    best = None
    for _ in range(int(n_starts)):
        order = rng.permutation(len(search.y))
        cand = search.evaluate(order)
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def initialize_state(
    osr,
    search,
    order_z_opt,
    start,
    seed,
    colony_size,
    use_guided_moves,
):
    rng = np.random.default_rng(seed + 100)
    random_rng = np.random.default_rng(seed + 200)

    population = [start.copy()]
    best = start.copy()
    evaluations_abc = 1

    while len(population) < colony_size:
        target_order = order_z_opt if use_guided_moves else None
        cand = search.evaluate(
            osr.mutate_order(
                rng,
                start["order"],
                target_order=target_order,
                guided_prob=0.30 if use_guided_moves else 0.0,
            )
        )
        evaluations_abc += 1
        population.append(cand)
        if cand["score"] > best["score"]:
            best = cand.copy()

    random_best = start.copy()
    evaluations_random = 1
    while evaluations_random < evaluations_abc:
        cand = search.evaluate(random_rng.permutation(len(search.y)))
        evaluations_random += 1
        if cand["score"] > random_best["score"]:
            random_best = cand.copy()

    return {
        "iteration": 0,
        "rng_state": rng.bit_generator.state,
        "random_rng_state": random_rng.bit_generator.state,
        "population": population,
        "best": best,
        "random_best": random_best,
        "evaluations_ABC": evaluations_abc,
        "evaluations_Random": evaluations_random,
    }


def checkpoint_row(
    iteration,
    z_var,
    pi_source,
    corr_y_z,
    corr_y_pi,
    corr_ratio,
    start,
    z_opt,
    y_opt,
    state,
):
    best = state["best"]
    random_best = state["random_best"]
    return {
        "iteration": iteration,
        "z_variable": z_var,
        "pi_source": pi_source,
        "corr_y_z": corr_y_z,
        "corr_y_pi": corr_y_pi,
        "corr_y_over_pi_z_over_pi": corr_ratio,
        "BestRandomStart_z_over_BestRandomStart_z": 1.0,
        "BestRandomStart_y_over_BestRandomStart_y": 1.0,
        "BestRandomStart_z_over_Vincent_z": ratio(start["eff_z"], z_opt["eff_z"]),
        "BestRandomStart_y_over_Vincent_y": ratio(start["eff_y"], y_opt["eff_y"]),
        "ABC_z_over_BestRandomStart_z": ratio(best["eff_z"], start["eff_z"]),
        "ABC_y_over_BestRandomStart_y": ratio(best["eff_y"], start["eff_y"]),
        "ABC_z_over_Vincent_z": ratio(best["eff_z"], z_opt["eff_z"]),
        "ABC_y_over_Vincent_y": ratio(best["eff_y"], y_opt["eff_y"]),
        "Random_z_over_BestRandomStart_z": ratio(random_best["eff_z"], start["eff_z"]),
        "Random_y_over_BestRandomStart_y": ratio(random_best["eff_y"], start["eff_y"]),
        "Random_z_over_Vincent_z": ratio(random_best["eff_z"], z_opt["eff_z"]),
        "Random_y_over_Vincent_y": ratio(random_best["eff_y"], y_opt["eff_y"]),
        "evaluations_ABC": state["evaluations_ABC"],
        "evaluations_Random": state["evaluations_Random"],
        "Var_z_BestRandomStart": start["var_z"],
        "Var_y_BestRandomStart": start["var_y"],
        "Var_z_Vincent_z": z_opt["var_z"],
        "Var_y_Vincent_y": y_opt["var_y"],
        "Var_z_ABC": best["var_z"],
        "Var_y_ABC": best["var_y"],
        "Var_z_Random": random_best["var_z"],
        "Var_y_Random": random_best["var_y"],
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
        random_starts,
        seed,
        resume,
        use_guided_moves,
    ) = args

    osr = load_order_runner()
    vd = osr.load_base()
    runner = vd.load_runner()
    df_pop, _z_names, _pi_sources = vd.make_simulated_population()

    y = df_pop["y"].to_numpy(dtype=float)
    z = df_pop[z_var].to_numpy(dtype=float)
    pi = vd.build_pi(runner, df_pop, pi_source)
    var_srs_y = vd.srs_variance(y)
    var_srs_z = vd.srs_variance(z)
    search = osr.OrderSearch(vd, runner, y, z, pi, var_srs_y, var_srs_z)

    order_z_opt = np.argsort(z / pi)
    order_y_opt = np.argsort(y / pi)
    z_opt = search.evaluate(order_z_opt)
    y_opt = search.evaluate(order_y_opt)

    corr_y_z = float(np.corrcoef(y, z)[0, 1])
    corr_y_pi = 0.0 if pi_source == "equal" else float(np.corrcoef(y, pi)[0, 1])
    corr_ratio = float(np.corrcoef(y / pi, z / pi)[0, 1])

    path = state_path(stem, z_var, pi_source)
    path.parent.mkdir(parents=True, exist_ok=True)

    if resume and path.exists():
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        start = saved["start"]
        state = saved["state"]
    else:
        start_rng = np.random.default_rng(seed)
        start = evaluate_best_random_start(search, start_rng, random_starts)
        state = initialize_state(
            osr,
            search,
            order_z_opt,
            start,
            seed,
            colony_size,
            use_guided_moves,
        )

    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    random_rng = np.random.default_rng()
    random_rng.bit_generator.state = state["random_rng_state"]

    checkpoint_rows = []
    current_iteration = int(state["iteration"])
    if current_iteration >= target_iterations:
        row = checkpoint_row(
            current_iteration,
            z_var,
            pi_source,
            corr_y_z,
            corr_y_pi,
            corr_ratio,
            start,
            z_opt,
            y_opt,
            state,
        )
        return row, [row], current_iteration

    for iteration in range(current_iteration + 1, target_iterations + 1):
        before = state["evaluations_ABC"]
        progress = iteration / max(1, target_iterations)
        guided_prob = 0.10 + 0.35 * progress if use_guided_moves else 0.0
        target_order = order_z_opt if use_guided_moves else None

        population = state["population"]
        best = state["best"]
        for i, food in enumerate(list(population)):
            cand = search.evaluate(
                osr.mutate_order(rng, food["order"], target_order=target_order, guided_prob=guided_prob)
            )
            state["evaluations_ABC"] += 1
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
            cand = search.evaluate(
                osr.mutate_order(
                    rng,
                    population[i]["order"],
                    target_order=target_order,
                    guided_prob=guided_prob,
                )
            )
            state["evaluations_ABC"] += 1
            if cand["score"] > population[i]["score"]:
                population[i] = cand
                if cand["score"] > best["score"]:
                    best = cand.copy()
            else:
                population[i]["trial"] += 1

        for i, food in enumerate(list(population)):
            if food["trial"] >= limit:
                base = best["order"] if rng.random() < 0.7 else start["order"]
                cand = search.evaluate(
                    osr.mutate_order(
                        rng,
                        base,
                        target_order=target_order,
                        guided_prob=0.55 if use_guided_moves else 0.0,
                    )
                )
                state["evaluations_ABC"] += 1
                population[i] = cand
                if cand["score"] > best["score"]:
                    best = cand.copy()

        state["best"] = best
        state["population"] = population

        new_abc_evals = state["evaluations_ABC"] - before
        random_best = state["random_best"]
        for _ in range(new_abc_evals):
            cand = search.evaluate(random_rng.permutation(len(y)))
            state["evaluations_Random"] += 1
            if cand["score"] > random_best["score"]:
                random_best = cand.copy()
        state["random_best"] = random_best
        state["iteration"] = iteration
        state["rng_state"] = rng.bit_generator.state
        state["random_rng_state"] = random_rng.bit_generator.state

        if iteration % checkpoint_interval == 0 or iteration == target_iterations:
            row = checkpoint_row(
                iteration,
                z_var,
                pi_source,
                corr_y_z,
                corr_y_pi,
                corr_ratio,
                start,
                z_opt,
                y_opt,
                state,
            )
            checkpoint_rows.append(row)
            with path.open("wb") as handle:
                pickle.dump({"start": start, "state": state}, handle)

    final = checkpoint_rows[-1]
    return final, checkpoint_rows, int(state["iteration"])


def parse_list(value):
    if value == "all":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--colony-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--onlooker-factor", type=float, default=0.5)
    parser.add_argument("--random-starts", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--z-vars", default="all")
    parser.add_argument("--pi-sources", default="all")
    args = parser.parse_args()

    osr = load_order_runner()
    vd = osr.load_base()
    _df, z_names, pi_sources = vd.make_simulated_population()
    selected_z = parse_list(args.z_vars) or z_names
    selected_pi = parse_list(args.pi_sources) or pi_sources

    mode = "guided" if args.guided else "unguided"
    stem = f"vincent_best_random_start_{mode}_{args.random_starts}starts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    final_path = ARTIFACT_DIR / f"{stem}_{args.iterations}iter.csv"
    checkpoint_path = ARTIFACT_DIR / f"{stem}_{args.iterations}iter_checkpoints.csv"

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
                    args.random_starts,
                    args.seed + 1000 * z_i + 17 * p_i,
                    args.resume,
                    args.guided,
                )
            )

    print("Best-random-Vincent-start experiment")
    print(f"cases              = {len(cases)}")
    print(f"random starts/case = {args.random_starts}")
    print(f"target iterations  = {args.iterations}")
    print(f"checkpoint interval= {args.checkpoint_interval}")
    print(f"resume             = {args.resume}")
    print(f"guided moves       = {args.guided}")
    print(f"final CSV          = {final_path}")
    print(f"checkpoint CSV     = {checkpoint_path}")
    print(f"state dir          = {STATE_DIR / stem}")

    start_time = time.time()
    finals = []
    checkpoints = []
    if args.workers <= 1:
        for case in cases:
            final, rows, current = run_case(case)
            finals.append(final)
            checkpoints.extend(rows)
            print(
                f"{len(finals)}/{len(cases)} {final['z_variable']} / {final['pi_source']} "
                f"iter={current}: ABC z/start={final['ABC_z_over_BestRandomStart_z']:.3f}, "
                f"ABC z/V={final['ABC_z_over_Vincent_z']:.3f}, "
                f"Rnd z/start={final['Random_z_over_BestRandomStart_z']:.3f}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_case, case): case for case in cases}
            for future in as_completed(future_map):
                final, rows, current = future.result()
                finals.append(final)
                checkpoints.extend(rows)
                print(
                    f"{len(finals)}/{len(cases)} {final['z_variable']} / {final['pi_source']} "
                    f"iter={current}: ABC z/start={final['ABC_z_over_BestRandomStart_z']:.3f}, "
                    f"ABC z/V={final['ABC_z_over_Vincent_z']:.3f}, "
                    f"Rnd z/start={final['Random_z_over_BestRandomStart_z']:.3f}"
                )

    final_df = pd.DataFrame(finals).sort_values(["z_variable", "pi_source"]).reset_index(drop=True)
    checkpoint_df = pd.DataFrame(checkpoints).sort_values(
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
        "BestRandomStart_z_over_Vincent_z",
        "BestRandomStart_y_over_Vincent_y",
        "ABC_z_over_BestRandomStart_z",
        "ABC_y_over_BestRandomStart_y",
        "ABC_z_over_Vincent_z",
        "ABC_y_over_Vincent_y",
        "Random_z_over_BestRandomStart_z",
        "Random_y_over_BestRandomStart_y",
        "Random_z_over_Vincent_z",
        "Random_y_over_Vincent_y",
        "evaluations_ABC",
        "evaluations_Random",
    ]
    print()
    print(final_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved final CSV: {final_path}")
    print(f"Saved checkpoint CSV: {checkpoint_path}")
    print(f"Elapsed: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
