"""Vincent-direct simulated 9-case experiment.

This runner follows the agreement with Bardia:
- build the exact Vincent/Ppi design in z/pi order;
- compute z and y HT variances from that z-optimal starting design;
- build a separate Vincent/Ppi benchmark in y/pi order for y;
- run ABC and random search from the same z-optimal design;
- give random search exactly the same number of evaluated designs as ABC.
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
ARTIFACT_DIR = PROJECT_ROOT / "simulations_abc" / "jupyters" / "artifacts"

N_UNITS = 50
N_SAMPLE = 5
SIMULATED_CORRELATIONS = [0.00, 0.80, 0.90]
PI_SIZE_STRENGTH = 0.35
BASE_SEED = 202600384

DIRECT_INIT_THETA_SCALE = 0.006
DIRECT_MUTATION_THETA_SCALE = 0.010
DIRECT_RANDOM_THETA_SCALE = 0.012
DIRECT_SCOUT_THETA_SCALE = 0.014
DIRECT_ROTATION_MAX_TRIES = 300
DIRECT_DIAG_TOL = 5e-8


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
    size_names = ["equal"]

    for corr in SIMULATED_CORRELATIONS:
        suffix = corr_suffix(corr)
        z_name = f"z_{suffix}"
        data[z_name] = correlated_with_y(y_std, corr, rng)
        z_names.append(z_name)

        if suffix != "00":
            size_name = f"size_{suffix}"
            data[size_name] = correlated_with_y(y_std, corr, rng)
            size_names.append(size_name)

    return pd.DataFrame(data), z_names, size_names


def make_size_for_pi(x):
    return np.exp(PI_SIZE_STRENGTH * standardize(x))


def build_pi(runner, df_pop, size_name):
    if size_name == "equal":
        return np.full(len(df_pop), N_SAMPLE / len(df_pop))
    size = make_size_for_pi(df_pop[size_name].to_numpy(dtype=float))
    return runner.inclusionprobabilities(size, N_SAMPLE)


def srs_variance(values, n_sample=N_SAMPLE):
    values = np.asarray(values, dtype=float)
    n_units = len(values)
    return n_units**2 * (1.0 - n_sample / n_units) * np.var(values, ddof=1) / n_sample


def symmetrize(K):
    K = np.asarray(K, dtype=np.complex128)
    return 0.5 * (K + K.conj().T)


def vincent_kernel(runner, pi_ordered):
    V = runner.Ppi(np.asarray(pi_ordered, dtype=float))
    return symmetrize(V @ V.T)


def ht_variance(K, pi_ordered, values_ordered):
    pi_ordered = np.asarray(pi_ordered, dtype=float)
    values_ordered = np.asarray(values_ordered, dtype=float)
    diag_K = np.real(np.diag(K))
    A = -np.abs(K) ** 2
    np.fill_diagonal(A, diag_K * (1.0 - diag_K))
    values_over_pi = values_ordered / pi_ordered
    return float(np.real(values_over_pi @ (A @ values_over_pi)))


def efficiency(var_srs, var_design):
    if var_design <= 0:
        return np.inf
    return float(var_srs / var_design)


def safe_ratio(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return float(value / reference)


def apply_diag_preserving_pair_rotation(K, i, j, theta, branch=0):
    K = np.asarray(K, dtype=np.complex128)
    a = float(np.real(K[i, i]))
    b = float(np.real(K[j, j]))
    x = K[i, j]
    absx = float(np.abs(x))
    diff = b - a

    if absx < 1e-14 and abs(diff) > 1e-14:
        return None

    if abs(diff) > 1e-14 and absx > 0:
        max_theta = np.arctan(2.0 * absx / abs(diff)) * 0.95
        theta = float(np.clip(theta, -max_theta, max_theta))

    if abs(theta) < 1e-14:
        return None

    c = float(np.cos(theta))
    s = float(np.sin(theta))

    if absx < 1e-14:
        psi = 0.0
    else:
        target = np.tan(theta) * diff / 2.0
        val = float(np.clip(target / absx, -1.0, 1.0))
        alpha = float(np.angle(x))
        acos_val = float(np.arccos(val))
        psi = (acos_val - alpha) if (branch % 2 == 0) else (-acos_val - alpha)

    U = np.eye(K.shape[0], dtype=np.complex128)
    epos = np.exp(1j * psi)
    eneg = np.exp(-1j * psi)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -s * eneg
    U[j, i] = s * epos
    return symmetrize(U @ K @ U.conj().T)


def local_rotate_kernel(rng, K, pi_target, theta_scale, n_steps=1):
    K_new = symmetrize(K)
    n_units = K_new.shape[0]
    accepted = 0

    for _ in range(max(1, int(n_steps))):
        for _try in range(DIRECT_ROTATION_MAX_TRIES):
            i, j = rng.choice(n_units, size=2, replace=False)
            if i > j:
                i, j = j, i
            if np.abs(K_new[i, j]) < 1e-12:
                continue

            theta = float(rng.normal(0.0, theta_scale))
            if abs(theta) < 1e-10:
                continue

            candidate = apply_diag_preserving_pair_rotation(
                K_new, i, j, theta, branch=int(rng.integers(0, 2))
            )
            if candidate is None:
                continue

            diag_error = float(np.max(np.abs(np.real(np.diag(candidate)) - pi_target)))
            if np.isfinite(diag_error) and diag_error <= DIRECT_DIAG_TOL:
                K_new = candidate
                accepted += 1
                break

    return K_new, accepted


class DirectSearchState:
    def __init__(self, K_seed, pi_z, y_z_order, z_z_order, var_srs_y, var_srs_z, rng):
        self.K_seed = symmetrize(K_seed)
        self.pi_z = np.asarray(pi_z, dtype=float)
        self.y_z_order = np.asarray(y_z_order, dtype=float)
        self.z_z_order = np.asarray(z_z_order, dtype=float)
        self.var_srs_y = float(var_srs_y)
        self.var_srs_z = float(var_srs_z)
        self.rng = rng
        self.evaluations = 0
        self.valid_evaluations = 0
        self.best_K = None
        self.best_var_z = np.inf
        self.best_var_y = np.inf
        self.best_eff_z = 0.0
        self.best_eff_y = 0.0

    def evaluate(self, K):
        self.evaluations += 1
        K = symmetrize(K)
        diag_error = float(np.max(np.abs(np.real(np.diag(K)) - self.pi_z)))
        if not np.isfinite(diag_error) or diag_error > DIRECT_DIAG_TOL:
            return None

        self.valid_evaluations += 1
        var_z = ht_variance(K, self.pi_z, self.z_z_order)
        var_y = ht_variance(K, self.pi_z, self.y_z_order)
        eff_z = efficiency(self.var_srs_z, var_z)
        eff_y = efficiency(self.var_srs_y, var_y)

        if var_z < self.best_var_z:
            self.best_K = K.copy()
            self.best_var_z = var_z
            self.best_var_y = var_y
            self.best_eff_z = eff_z
            self.best_eff_y = eff_y

        return {
            "K": K,
            "var_z": var_z,
            "var_y": var_y,
            "eff_z": eff_z,
            "eff_y": eff_y,
            "score": eff_z,
            "trial": 0,
        }

    def random_candidate(self, base, theta_scale, n_steps=1):
        K_new, _accepted = local_rotate_kernel(
            self.rng,
            base,
            self.pi_z,
            theta_scale=theta_scale,
            n_steps=n_steps,
        )
        return self.evaluate(K_new)


class DirectABC:
    def __init__(
        self,
        K_seed,
        pi_z,
        y_z_order,
        z_z_order,
        var_srs_y,
        var_srs_z,
        seed,
        colony_size=20,
        limit=5,
        onlooker_factor=0.5,
    ):
        self.rng = np.random.default_rng(seed)
        self.state = DirectSearchState(K_seed, pi_z, y_z_order, z_z_order, var_srs_y, var_srs_z, self.rng)
        self.colony_size = int(colony_size)
        self.limit = int(limit)
        self.onlooker_count = int(round(self.colony_size * float(onlooker_factor)))
        self.population = []

    @property
    def evaluations(self):
        return self.state.evaluations

    @property
    def best_K(self):
        return self.state.best_K

    @property
    def best_eff_z(self):
        return self.state.best_eff_z

    @property
    def best_eff_y(self):
        return self.state.best_eff_y

    @property
    def best_var_z(self):
        return self.state.best_var_z

    @property
    def best_var_y(self):
        return self.state.best_var_y

    def initialize(self):
        seed_food = self.state.evaluate(self.state.K_seed)
        if seed_food is None:
            raise RuntimeError("Vincent z-optimal seed failed validation.")
        self.population = [seed_food]
        while len(self.population) < self.colony_size:
            food = self.state.random_candidate(
                self.state.K_seed,
                theta_scale=DIRECT_INIT_THETA_SCALE,
                n_steps=1,
            )
            if food is not None:
                self.population.append(food)

    def _replace_if_better(self, index, new_food):
        old_food = self.population[index]
        if new_food is not None and new_food["score"] > old_food["score"]:
            self.population[index] = new_food
        else:
            old_food["trial"] += 1

    def step(self, iteration, max_iterations):
        before = self.evaluations
        progress = iteration / max(1, max_iterations)
        theta_scale = DIRECT_MUTATION_THETA_SCALE * max(0.25, 1.0 - 0.6 * progress)

        for i, food in enumerate(list(self.population)):
            new_food = self.state.random_candidate(
                food["K"],
                theta_scale=theta_scale,
                n_steps=1,
            )
            self._replace_if_better(i, new_food)

        scores = np.array([max(0.0, f["score"]) for f in self.population], dtype=float)
        if scores.sum() <= 0:
            probabilities = np.full(len(scores), 1.0 / len(scores))
        else:
            probabilities = scores / scores.sum()

        for _ in range(self.onlooker_count):
            i = int(self.rng.choice(len(self.population), p=probabilities))
            new_food = self.state.random_candidate(
                self.population[i]["K"],
                theta_scale=theta_scale,
                n_steps=1,
            )
            self._replace_if_better(i, new_food)

        for i, food in enumerate(list(self.population)):
            if food["trial"] >= self.limit:
                base = self.best_K if self.best_K is not None else self.state.K_seed
                new_food = self.state.random_candidate(
                    base,
                    theta_scale=DIRECT_SCOUT_THETA_SCALE,
                    n_steps=2,
                )
                if new_food is not None:
                    self.population[i] = new_food

        return self.evaluations - before


class DirectRandom:
    def __init__(self, K_seed, pi_z, y_z_order, z_z_order, var_srs_y, var_srs_z, seed):
        self.rng = np.random.default_rng(seed)
        self.state = DirectSearchState(K_seed, pi_z, y_z_order, z_z_order, var_srs_y, var_srs_z, self.rng)

    @property
    def evaluations(self):
        return self.state.evaluations

    @property
    def best_eff_z(self):
        return self.state.best_eff_z

    @property
    def best_eff_y(self):
        return self.state.best_eff_y

    @property
    def best_var_z(self):
        return self.state.best_var_z

    @property
    def best_var_y(self):
        return self.state.best_var_y

    def step(self, n_evaluations):
        for i in range(int(n_evaluations)):
            if self.evaluations == 0 and i == 0:
                self.state.evaluate(self.state.K_seed)
            else:
                base = self.state.K_seed
                self.state.random_candidate(base, theta_scale=DIRECT_RANDOM_THETA_SCALE, n_steps=2)


def run_case(args):
    z_var, size_var, iterations, checkpoint_interval, colony_size, limit, onlooker_factor = args
    runner = load_runner()
    df_pop, _z_names, _size_names = make_simulated_population()

    y_raw = df_pop["y"].to_numpy(dtype=float)
    z_raw = df_pop[z_var].to_numpy(dtype=float)
    pi_raw = build_pi(runner, df_pop, size_var)

    order_z = np.argsort(z_raw / pi_raw)
    order_y = np.argsort(y_raw / pi_raw)

    y_z_order = y_raw[order_z]
    z_z_order = z_raw[order_z]
    pi_z_order = pi_raw[order_z]

    y_y_order = y_raw[order_y]
    pi_y_order = pi_raw[order_y]

    K_z_vincent = vincent_kernel(runner, pi_z_order)
    K_y_vincent = vincent_kernel(runner, pi_y_order)

    var_srs_y = srs_variance(y_raw)
    var_srs_z = srs_variance(z_raw)

    start_var_z = ht_variance(K_z_vincent, pi_z_order, z_z_order)
    start_var_y = ht_variance(K_z_vincent, pi_z_order, y_z_order)
    y_vincent_var_y = ht_variance(K_y_vincent, pi_y_order, y_y_order)

    start_eff_z = efficiency(var_srs_z, start_var_z)
    start_eff_y = efficiency(var_srs_y, start_var_y)
    y_vincent_eff_y = efficiency(var_srs_y, y_vincent_var_y)

    corr_y_z = float(np.corrcoef(y_raw, z_raw)[0, 1])
    corr_y_pi = 0.0 if size_var == "equal" else float(np.corrcoef(y_raw, pi_raw)[0, 1])
    corr_ratio = float(np.corrcoef(y_raw / pi_raw, z_raw / pi_raw)[0, 1])

    case_seed = BASE_SEED + 1000 * (int(z_var.split("_")[1]) + 1) + 10 * (
        0 if size_var == "equal" else int(size_var.split("_")[1])
    )
    abc = DirectABC(
        K_z_vincent,
        pi_z_order,
        y_z_order,
        z_z_order,
        var_srs_y,
        var_srs_z,
        seed=case_seed + 1,
        colony_size=colony_size,
        limit=limit,
        onlooker_factor=onlooker_factor,
    )
    rnd = DirectRandom(
        K_z_vincent,
        pi_z_order,
        y_z_order,
        z_z_order,
        var_srs_y,
        var_srs_z,
        seed=case_seed + 2,
    )

    abc.initialize()
    rnd.step(abc.evaluations)

    checkpoints = []
    for iteration in range(1, iterations + 1):
        new_evals = abc.step(iteration, iterations)
        rnd.step(new_evals)

        if iteration % checkpoint_interval == 0 or iteration == iterations:
            checkpoints.append(
                {
                    "iteration": iteration,
                    "z_variable": z_var,
                    "pi_source": size_var,
                    "corr_y_z": corr_y_z,
                    "corr_y_pi": corr_y_pi,
                    "corr_y_over_pi_z_over_pi": corr_ratio,
                    "Start_z_over_Vincent_z": safe_ratio(start_eff_z, start_eff_z),
                    "Start_y_over_Vincent_y": safe_ratio(start_eff_y, y_vincent_eff_y),
                    "ABC_z_over_Vincent_z": safe_ratio(abc.best_eff_z, start_eff_z),
                    "ABC_y_over_Vincent_y": safe_ratio(abc.best_eff_y, y_vincent_eff_y),
                    "Random_z_over_Vincent_z": safe_ratio(rnd.best_eff_z, start_eff_z),
                    "Random_y_over_Vincent_y": safe_ratio(rnd.best_eff_y, y_vincent_eff_y),
                    "evaluations_ABC": abc.evaluations,
                    "evaluations_Random": rnd.evaluations,
                }
            )

    final = checkpoints[-1].copy()
    final.update(
        {
            "Var_z_Vincent_z": start_var_z,
            "Var_y_Vincent_z": start_var_y,
            "Var_y_Vincent_y": y_vincent_var_y,
            "Var_z_ABC": abc.best_var_z,
            "Var_y_ABC": abc.best_var_y,
            "Var_z_Random": rnd.best_var_z,
            "Var_y_Random": rnd.best_var_y,
        }
    )
    return final, checkpoints


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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--z-vars", default="all")
    parser.add_argument("--pi-sources", default="all")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df_pop, z_names, size_names = make_simulated_population()
    selected_z = parse_list(args.z_vars) or z_names
    selected_sizes = parse_list(args.pi_sources) or size_names

    cases = [
        (
            z_var,
            size_var,
            args.iterations,
            args.checkpoint_interval,
            args.colony_size,
            args.limit,
            args.onlooker_factor,
        )
        for z_var in selected_z
        for size_var in selected_sizes
    ]

    stamp = f"vincent_direct_abc_random_9cases_{args.iterations}iter"
    final_path = ARTIFACT_DIR / f"{stamp}.csv"
    checkpoint_path = ARTIFACT_DIR / f"{stamp}_checkpoints.csv"
    tex_path = ARTIFACT_DIR / f"{stamp}_table.tex"

    print("Vincent-direct simulated experiment")
    print(f"cases              = {len(cases)}")
    print(f"iterations         = {args.iterations}")
    print(f"checkpoint interval= {args.checkpoint_interval}")
    print(f"colony size        = {args.colony_size}")
    print(f"workers            = {args.workers}")
    print(f"final CSV          = {final_path}")
    print(f"checkpoint CSV     = {checkpoint_path}")

    start = time.time()
    finals = []
    checkpoint_rows = []
    if args.workers <= 1 or len(cases) == 1:
        for case in cases:
            final, checkpoints = run_case(case)
            finals.append(final)
            checkpoint_rows.extend(checkpoints)
            print(
                f"{len(finals)}/{len(cases)} {final['z_variable']} / {final['pi_source']}: "
                f"ABC z={final['ABC_z_over_Vincent_z']:.4f}, "
                f"ABC y={final['ABC_y_over_Vincent_y']:.4f}, "
                f"Rnd z={final['Random_z_over_Vincent_z']:.4f}, "
                f"Rnd y={final['Random_y_over_Vincent_y']:.4f}, "
                f"evals={final['evaluations_ABC']}"
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
                    f"ABC z={final['ABC_z_over_Vincent_z']:.4f}, "
                    f"ABC y={final['ABC_y_over_Vincent_y']:.4f}, "
                    f"Rnd z={final['Random_z_over_Vincent_z']:.4f}, "
                    f"Rnd y={final['Random_y_over_Vincent_y']:.4f}, "
                    f"evals={final['evaluations_ABC']}"
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
    ]
    table = final_df[display_cols].copy()
    table.to_latex(tex_path, index=False, float_format="%.4f")

    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved final table: {final_path}")
    print(f"Saved checkpoints: {checkpoint_path}")
    print(f"Saved LaTeX table: {tex_path}")
    print(f"Total elapsed: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
