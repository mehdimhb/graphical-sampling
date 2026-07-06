#!/usr/bin/env python3
"""
Small terminal reproduction of Vincent's omega/rho warm-start example.

Run from the repository root:
    .venv/bin/python simulations_abc/terminal_running/vincent_three_case_check.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np


RUNNER_PATH = Path(__file__).with_name("run_abc_mu284_terminal.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_abc_mu284_terminal", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def kernel_quadratic_form(K: np.ndarray, y_over_pi: np.ndarray) -> float:
    diag_k = np.real(np.diag(K))
    A = -np.abs(K) ** 2
    np.fill_diagonal(A, diag_k * (1.0 - diag_k))
    return float(np.real(y_over_pi @ (A @ y_over_pi)))


def make_vincent_population(seed: int = 123, N: int = 50, S: int = 7):
    rng = np.random.default_rng(seed)
    pi = rng.uniform(0.4, 0.8, N)
    pi = pi * S / pi.sum()
    pi = np.sort(pi)[::-1]

    y1 = 5.0
    ratio_increments = np.concatenate(([y1 / pi[0]], rng.uniform(0.2, 1.0, N - 1)))
    y_over_pi = np.cumsum(ratio_increments)
    y = y_over_pi * pi

    if not np.all(np.diff(y / pi) > 0):
        raise RuntimeError("The generated y/pi ratios are not increasing.")

    return pi, y, y_over_pi


def main() -> None:
    runner = _load_runner()
    N = 50
    S = 7
    pi, _, y_over_pi = make_vincent_population(N=N, S=S)

    Vopt = runner.Ppi(pi)
    Kopt = Vopt @ Vopt.T
    v0 = kernel_quadratic_form(Kopt, y_over_pi)

    cases = [
        ("omega=0, rho=0", 0.0, 0.0),
        ("omega=0, rho=0.5", 0.0, 0.5),
        ("omega=0.5, rho=0.5", 0.5, 0.5),
    ]

    print("Vincent warm-start variance check")
    print(f"N={N}, S={S}, increasing y/pi={np.all(np.diff(y_over_pi) > 0)}")
    print(f"Ppi variance v0 = {v0:.12g}")
    print()
    print(f"{'case':<24} {'CaDsd variance':>18} {'v1/v0':>12}")
    print("-" * 58)

    for label, omega_value, rho_value in cases:
        omega = omega_value * np.ones((S, N))
        rho = rho_value * np.ones((S, N - 1))
        K = runner.CaDsd(pi=pi, M=S, omega=omega, rho=rho)["K"]
        v1 = kernel_quadratic_form(K, y_over_pi)
        print(f"{label:<24} {v1:18.12g} {v1 / v0:12.6g}")


if __name__ == "__main__":
    main()
