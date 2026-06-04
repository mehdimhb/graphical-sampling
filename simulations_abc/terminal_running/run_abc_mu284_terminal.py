#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terminal script for ABC + Random Search on MU284.
Generated from Vincent_Sampling_abc_fast_random.ipynb, with clean user settings.
"""

import numpy as np
from typing import Union, Optional


# ------------------------------------------------------------
# 1) Ppi: build the orthogonal matrix V from first-order Pis
# ------------------------------------------------------------

def Ppi(Pi: Union[np.ndarray, list]) -> np.ndarray:
    """
    Python/Numpy port of the R function Ppi(Pi).

    Parameters
    ----------
    Pi : array-like of shape (N,)
        First-order inclusion probabilities, 0 < Pi_i < 1,
        sum(Pi) must be (approximately) an integer.

    Returns
    -------
    V : ndarray of shape (N, n)
        Orthogonal matrix associated to the DSD construction.
    """
    Pi = np.asarray(Pi, dtype=float).ravel()
    N = Pi.size

    # --- Error checks ---
    if N < 2:
        raise ValueError(
            "The sampling designs should be defined on a set of more than "
            "one element. (length(Pi) > 1)"
        )

    if np.any(Pi <= 0) or np.any(Pi >= 1):
        raise ValueError("Pi is not a vector of probabilities (0 < p < 1).")

    sum_pi_rounded = round(Pi.sum(), 9)
    n_int = int(sum_pi_rounded)
    if int(round(sum_pi_rounded, 9)) - sum_pi_rounded != 0:
        raise ValueError(
            "The sum of the first order inclusion probabilities "
            "should be an integer (up to rounding)."
        )

    # --- Main algorithm ---
    s_vals = np.zeros(N, dtype=float)
    c_vals = np.zeros(N, dtype=float)
    alpha = np.zeros(N, dtype=float)

    # kr will store the indices (0-based) where cumulative sum crosses integers
    if n_int <= 0:
        raise ValueError("Sum of Pi must be at least 1.")
    kr = [None] * n_int

    cum_sum = 0.0
    r = 1          # current integer threshold
    r_prev = 0     # last integer that was crossed

    for k in range(N):
        prev_sum = cum_sum
        cum_sum += Pi[k]

        if cum_sum >= r:  # crossed integer r
            if r <= n_int:
                alpha[k] = r - prev_sum
                kr[r - 1] = k   # store 0-based index
                val = np.sqrt((1.0 - Pi[k]) / (1.0 - alpha[k]))
                s_vals[k] = np.round(val, 8)
                r_prev = r
                r += 1
        else:
            denom = (r_prev + 1 - prev_sum)
            val = np.sqrt(Pi[k] / denom)
            s_vals[k] = np.round(val, 15)

        c_vals[k] = np.sqrt(1.0 - s_vals[k] ** 2)

    # Patch: ensure last crossing index corresponds to last unit (like R hack)
    # If some entries of kr are still None, set the last one to N-1.
    if any(x is None for x in kr):
        kr[-1] = N - 1
    # For safety, also replace any remaining None by the last index
    last_index = kr[-1]
    kr = [last_index if x is None else x for x in kr]

    # Use sample size n_int as number of columns (simplified vs. R hack)
    r_prev = n_int

    # --- Build V ---
    V = np.zeros((N, r_prev), dtype=float)
    V[0, 0] = 1.0

    # In R: V[kr[r] + 1, r + 1] = 1 for r in 1:(r_prev-1)
    # Here r_idx corresponds to r-1 in R, and we use 0-based indices.
    if r_prev - 1 != 0:
        for r_idx in range(1, r_prev):
            kpos = kr[r_idx - 1]  # 0-based index for row
            V[kpos + 1, r_idx] = 1.0

    # Apply Givens-like rotations
    for k in range(N - 1):
        L = V[k, :].copy()
        M = V[k + 1, :].copy()
        V[k, :] = s_vals[k] * L - c_vals[k] * M
        V[k + 1, :] = c_vals[k] * L + s_vals[k] * M

    return V


# ------------------------------------------------------------
# 2) DSD sampling (real and complex versions)
# ------------------------------------------------------------

def Drawing_Dsd(
    v: Union[np.ndarray, list],
    s: int = 1,
    B: bool = False,
    seed: Optional[int] = None,
):
    """
    Python/Numpy port of the R function Drawing_Dsd(v, s=1, B=FALSE, seed=NULL).

    Parameters
    ----------
    v : array-like, shape (N, n)
        Matrix of (real or complex) vectors used in DSD.
    s : int, default 1
        Number of samples (replicates).
    B : bool, default False
        If True: return 0/1 indicator vector(s).
        If False: return indices of selected units (1-based, to match R).
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    If s == 1:
        1D array of length N (if B=True) or selected indices (if B=False).
    If s > 1:
        2D array of shape (N, s) (if B=True) or (n, s) with indices per sample.
    """
    v = np.asarray(v)
    rng = np.random.default_rng(seed)

    if np.iscomplexobj(v):
        return _dsd_sampling_mult_complex(v, s, B, rng)
    else:
        return _dsd_sampling_mult(v, s, B, rng)


# ---------------------- helpers: real case ---------------------- #

def _dsd_sampling_mult(
    v: np.ndarray,
    s: int,
    B: bool,
    rng: np.random.Generator,
):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v[:, None]  # treat as (N,1)

    if s == 1:
        return _dsd_sampling_01_B_C(v, B, rng)
    else:
        samples = [_dsd_sampling_01_B_C(v, B, rng) for _ in range(s)]
        # stack as columns like R's replicate (N x s)
        return np.column_stack(samples)


def _dsd_sampling_01_B_C(
    v: np.ndarray,
    B: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """
    Real-valued version of .dsd_sampling_01_B_C in R.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v[:, None]

    N, n = v.shape
    echant = np.zeros(N, dtype=int)

    if rng is None:
        rng = np.random.default_rng()
    ref = rng.random(n)

    # Step 1: first element
    w = v.copy()

    pi1 = np.einsum("ij,ij->i", v, v)  # diag(v %*% t(v))
    total = 0.0
    i = -1

    while total < ref[0]:
        i += 1
        if i >= N:
            raise RuntimeError("Sampling failed in first step (real case).")
        total += pi1[i] / n
    echant[i] = 1

    l = v[i, :]
    norm_l = np.sqrt(np.dot(l, l))
    if norm_l == 0:
        raise ValueError("Encountered zero-norm vector in real DSD.")
    e1 = l / norm_l

    # Step 2: remaining n-1 elements
    for j in range(n - 1):
        r = n - (j + 1)
        inter = v @ e1  # (N,)
        pi1 = pi1 - inter * inter
        pi2 = pi1 / r

        total = 0.0
        i = -1
        while total < ref[j + 1]:
            i += 1
            if i >= N:
                raise RuntimeError("Sampling failed in step 2 (real case).")
            total += pi2[i]
        echant[i] = 1

        # Update w and e1 (Gram-Schmidt-like update)
        proj = w @ e1                      # shape (N,)
        w = w - np.outer(proj, e1)         # (N,n)
        L = w[i, :]
        norm_L = np.sqrt(np.dot(L, L))
        if norm_L == 0:
            raise ValueError("Encountered zero-norm vector in real DSD.")
        e1 = L / norm_L

    if B:
        # 0/1 vector -> same as R "echant"
        return echant
    else:
        # indices (1-based, like in R: (1:N)[echant==1])
        return np.nonzero(echant == 1)[0] + 1


# -------------------- helpers: complex case --------------------- #

def _dsd_sampling_mult_complex(
    v: np.ndarray,
    s: int,
    B: bool,
    rng: np.random.Generator,
):
    v = np.asarray(v, dtype=np.complex128)
    if v.ndim == 1:
        v = v[:, None]

    if s == 1:
        return _dsd_sampling_01_B_C_complex(v, B, rng)
    else:
        samples = [_dsd_sampling_01_B_C_complex(v, B, rng) for _ in range(s)]
        return np.column_stack(samples)


def _dsd_sampling_01_B_C_complex(
    v: np.ndarray,
    B: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """
    Complex-valued version of .dsd_sampling_01_B_C_complex in R.
    """
    v = np.asarray(v, dtype=np.complex128)
    if v.ndim == 1:
        v = v[:, None]

    N, n = v.shape
    echant = np.zeros(N, dtype=int)

    if rng is None:
        rng = np.random.default_rng()
    ref = rng.random(n)

    w = v.copy()

    # pi1 = Re( diag( v %*% t(Conj(v)) ) )
    pi1 = np.real(np.einsum("ij,ij->i", v, np.conjugate(v)))

    if np.any(pi1 < 0) or np.any(pi1 >= 1):
        raise ValueError(
            "The matrix v given as input doesn't suit the expected input "
            "(cf. pgd / periodic_dsd)."
        )

    # Step 1: first element
    total = 0.0
    i = -1

    while total < ref[0]:
        i += 1
        if i >= N:
            raise RuntimeError("Sampling failed in first step (complex case).")
        total += pi1[i] / n
    echant[i] = 1

    M = v[i, :]
    norm_M = np.sqrt(np.real(np.vdot(M, M)))
    if norm_M == 0:
        raise ValueError("Encountered zero-norm vector in complex DSD.")
    e1 = M / norm_M

    # Step 2: remaining n-1 elements
    for j in range(n - 1):
        r = n - (j + 1)

        # inter <- v %*% Conj(e1)
        inter = v @ np.conjugate(e1)           # (N,)
        pi1 = pi1 - np.real(inter * np.conjugate(inter))
        pi2 = np.real(pi1 / r)

        total = 0.0
        i = -1
        while total < ref[j + 1]:
            i += 1
            if i >= N:
                raise RuntimeError("Sampling failed in step 2 (complex case).")
            total += pi2[i]
        echant[i] = 1

        # w <- w - (projection on e1)
        proj = w @ np.conjugate(e1)           # (N,)
        w = w - np.outer(proj, e1)            # (N, n)

        L = w[i, :]
        norm_L = np.sqrt(np.real(np.vdot(L, L)))
        if norm_L == 0:
            raise ValueError("Encountered zero-norm vector in complex DSD.")
        e1 = L / norm_L

    if B:
        return echant
    else:
        # Return 1-based indices, to match the R function
        return np.nonzero(echant == 1)[0] + 1

import numpy as np
from typing import Optional, Union


def spec(omega: np.ndarray, M: int, pi: Union[np.ndarray, list], spectre=100) -> np.ndarray:
    """
    Faster Python/Numpy port of the R function spec().

    The output is the same object as before: a matrix of shape (M, N).
    Speed-up comes from cumulative sums instead of thousands of repeated
    small numpy sum calls inside Python loops.
    """
    pi = np.asarray(pi, dtype=float).ravel()
    N = pi.size
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M, N):
        raise ValueError("omega must have shape (M, N)")

    mat_spectre = np.zeros((M, N), dtype=float)
    pi_down = np.sort(pi)[::-1]
    pi_prefix = np.concatenate(([0.0], np.cumsum(pi_down)))
    mu = float(pi.sum())

    if (np.isscalar(spectre) and spectre == 100) or (
        not np.isscalar(spectre)
        and len(np.atleast_1d(spectre)) == 1
        and np.atleast_1d(spectre)[0] == 100
    ):
        spectre_vec = np.zeros(M, dtype=float)
        cumsum_1 = 0.0
        lam = 0.0

        for j in range(M - 1):
            jR = j + 1
            A = max(lam, mu - cumsum_1 - (M - jR))
            B = 1.0

            for iR in range(1, M - jR + 1):
                t_idx = M - jR - iR + 1
                s_down = pi_prefix[t_idx]
                val = (mu - cumsum_1 - s_down) / iR
                if val < B:
                    B = val

            val = (mu - cumsum_1) / (M - jR + 1)
            if val < B:
                B = val

            spectre_vec[j] = A + omega[j, N - 1] * (B - A)
            lam = spectre_vec[j]
            cumsum_1 += spectre_vec[j]

        spectre_vec[M - 1] = mu - spectre_vec[: M - 1].sum()
    else:
        spectre_arr = np.asarray(spectre, dtype=float).ravel()
        if spectre_arr.size != M:
            raise ValueError("spectre must have length M")
        spectre_vec = spectre_arr

    mat_spectre[:, N - 1] = spectre_vec

    for kR in range(N - 1, 0, -1):
        startR = max(1, M - kR + 1)
        lambda1 = mat_spectre[:, kR].copy()
        lambda2 = mat_spectre[:, kR - 1].copy()
        prefix_l1 = np.concatenate(([0.0], np.cumsum(lambda1)))
        cum_l2_before_j = 0.0

        # rows before startR are zero by construction, so cum_l2_before_j starts at 0.
        for jR in range(startR, M + 1):
            lam_prev = lambda1[jR - 2] if jR >= 2 else 0.0
            sum_l1 = prefix_l1[jR]
            A = max(0.0, lam_prev, sum_l1 - cum_l2_before_j - pi_down[kR])

            B_inner = float("inf")
            for iR in range(jR, M + 1):
                start_idx0 = M - iR
                if start_idx0 < kR:
                    prem = pi_prefix[kR] - pi_prefix[start_idx0]
                else:
                    prem = 0.0

                if jR <= (iR - 1):
                    deux = prefix_l1[iR - 1] - prefix_l1[jR - 1]
                else:
                    deux = 0.0

                B_val = prem - deux - cum_l2_before_j
                if B_val < B_inner:
                    B_inner = B_val

            B = min(lambda1[jR - 1], B_inner)
            new_val = A + omega[jR - 1, kR - 1] * (B - A)
            mat_spectre[jR - 1, kR - 1] = new_val
            lambda2[jR - 1] = new_val
            cum_l2_before_j += new_val

    return mat_spectre


def CaDsd(
    pi: Union[np.ndarray, list],
    M: Optional[int] = None,
    omega: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    spectre=100,
    U: Optional[np.ndarray] = None,
    option: bool = True,
):
    """
    Faster CaDsd implementation with the same interface and output keys.

    Main speed-ups:
    - uses the faster spec() above;
    - preallocates phi instead of repeated hstack;
    - replaces permutation matrices by direct column indexing;
    - replaces multiplication by a diagonal phase matrix with column scaling.
    """
    pi = np.asarray(pi, dtype=float).ravel()
    N = pi.size

    if M is None:
        M = int(round(pi.sum(), 7))
    if int(M) != M:
        raise ValueError("M should be an integer")
    M = int(M)

    if omega is None:
        omega = 0.5 * np.ones((M, N), dtype=float)
    else:
        omega = np.asarray(omega, dtype=float)
        if omega.shape != (M, N):
            raise ValueError("omega must have shape (M, N)")

    if rho is None:
        rho = 0.5 * np.ones((M, N - 1), dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)
        if rho.shape != (M, N - 1):
            raise ValueError("rho must have shape (M, N-1)")

    rho_angles = np.round(rho * 2 * np.pi, 7)
    mat_spectre = np.round(spec(omega, M, pi, spectre), 7)
    pi_down = np.sort(pi)[::-1]

    if U is None:
        U_work = np.eye(M, dtype=complex)
    else:
        U_work = np.asarray(U, dtype=complex).copy()
        if U_work.shape != (M, M):
            raise ValueError("U must have shape (M, M)")

    phi = np.zeros((M, N), dtype=complex)
    phi[:, 0] = np.round(np.sqrt(pi_down[0]) * U_work[:, 0], 7)
    ens = np.arange(1, M + 1, dtype=int)

    eye_index = np.arange(M)

    for kR in range(2, N + 1):
        phases = np.exp(1j * rho_angles[:, kR - 2])
        lambda1 = mat_spectre[:, kR - 1].copy()
        lambda2 = mat_spectre[:, kR - 2].copy()

        E1 = ens.tolist()
        E2 = ens.tolist()

        # Keep exact equality because mat_spectre is rounded to 7 digits as in the earlier code.
        for jR in ens:
            if not E1:
                break
            val = lambda2[jR - 1]
            lam1_E1 = lambda1[np.array(E1) - 1]
            matches = np.where(lam1_E1 == val)[0]
            if matches.size > 0:
                E2 = [x for x in E2 if x != jR]
                del E1[matches[0]]

        E1_arr = np.array(E1, dtype=int)
        E2_arr = np.array(E2, dtype=int)
        E1_rev = M + 1 - E1_arr
        E2_rev = M + 1 - E2_arr
        r = len(E1_rev)

        if r == 0:
            continue

        if r != M:
            mask1 = np.ones(M, dtype=bool)
            mask2 = np.ones(M, dtype=bool)
            mask1[E1_rev - 1] = False
            mask2[E2_rev - 1] = False
            E1_perm = np.concatenate([np.sort(E1_rev), ens[mask1]])
            E2_perm = np.concatenate([np.sort(E2_rev), ens[mask2]])
            perm1 = E1_perm - 1
            perm2 = E2_perm - 1
        else:
            perm1 = eye_index
            perm2 = eye_index

        lambda2_E2 = lambda2[E2_arr - 1]
        lambda1_E1 = lambda1[E1_arr - 1]
        R = np.column_stack([lambda2_E2, lambda1_E1]).astype(complex)[::-1, :]

        v = np.zeros(r, dtype=complex)
        w = np.zeros(r, dtype=complex)

        for i in range(r):
            v1 = R[i, 0] - R[:, 1]
            v2 = R[i, 0] - R[:, 0]
            v2[i] = 1.0 + 0j
            order = np.argsort(np.abs(v1))
            v[i] = np.round(np.sqrt(-np.prod(v1[order] / v2[order])), 7)

            w1 = R[i, 1] - R[:, 0]
            w2 = R[i, 1] - R[:, 1]
            w2[i] = 1.0 + 0j
            order_w = np.argsort(np.abs(w1))
            w[i] = np.round(np.sqrt(np.prod(w1[order_w] / w2[order_w])), 7)

        col_vals = R[:, 1]
        row_vals = R[:, 0]
        denom = col_vals[np.newaxis, :] - row_vals[:, np.newaxis]
        W = (v[:, None] * w[None, :]) / denom

        # U @ diag(phases) is just column scaling.
        UV = U_work * phases[np.newaxis, :]

        # Equivalent to U %*% V %*% t(sigma2) %*% vect.
        UV_sigma2 = UV[:, perm2]
        phi[:, kR - 1] = UV_sigma2[:, :r] @ v

        # Equivalent to U %*% V %*% t(sigma2) %*% block %*% sigma1.
        U_block = UV_sigma2.copy()
        U_block[:, :r] = UV_sigma2[:, :r] @ W
        U_new = np.empty_like(U_block)
        U_new[:, perm1] = U_block
        U_work = U_new

    d = np.sqrt(mat_spectre[:, N - 1])
    if np.any(d == 0):
        raise ValueError("Zero on diagonal of spectrum, cannot invert sqrt.")

    # Avoid forming an explicit inverse diagonal matrix.
    EigenBasis = (phi.conj().T @ U_work) / d[np.newaxis, :]
    K = np.round(phi.conj().T @ phi, 7)

    return {
        "K": K,
        "spectrum": mat_spectre,
        "EigenBasis": EigenBasis,
    }


print("Fast spec() and CaDsd() loaded.")


import numpy as np

# --------------------------------------------------------
# 1) inclusionprobabilities() – Python version
#    (size measure p -> inclusion probs π with sum π = n)
# --------------------------------------------------------

def inclusionprobabilities(p, n):
    """
    Rough equivalent of sampling::inclusionprobabilities(p, n) in R.

    p : array-like, strictly positive size measures
    n : desired fixed sample size (integer)

    Returns
    -------
    pik : ndarray of length N
        First-order inclusion probabilities, 0 < pik_i <= 1, sum pik_i = n.
    """
    p = np.asarray(p, dtype=float)
    N = p.size
    if n <= 0 or n > N:
        raise ValueError("n must be in 1..N")

    pik = np.zeros(N, dtype=float)
    mask_fixed = np.zeros(N, dtype=bool)  # True where pik is already fixed to 1
    n_rem = float(n)
    p_work = p.copy()

    # Iterative rescaling: set any prob >= 1 to 1, rescale remaining
    while True:
        idx = ~mask_fixed
        if not np.any(idx):
            break

        p_sub = p_work[idx]
        # if all remaining p_sub are zero but n_rem > 0, it's impossible
        if p_sub.sum() <= 0 and n_rem > 1e-12:
            raise RuntimeError("Cannot construct inclusion probabilities with given p and n.")

        temp = n_rem * p_sub / p_sub.sum()
        over = temp >= (1.0 - 1e-12)  # tolerance

        # If no temp >= 1, we're done
        if not np.any(over):
            pik[idx] = temp
            break

        # Fix those >=1 to exactly 1
        idx_global = np.where(idx)[0]
        over_global = idx_global[over]

        pik[over_global] = 1.0
        mask_fixed[over_global] = True
        n_rem -= over.sum()
        p_work[over_global] = 0.0

        if n_rem <= 1e-12:
            # All remaining inclusion probabilities must be zero
            break

    return pik




import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any


class ABCAlgorithm:
    """
    Faster Artificial Bee Colony optimizer for the CaDsd parameterization.

    Speed changes that keep the method intact:
    - variance is computed with precomputed y/pi and z/pi vectors, no dense Dpi multiplications;
    - eigenvalue checks are optional/occasional because CaDsd should return a valid DSD kernel by construction;
    - early stopping is available but conservative;
    - onlooker intensity can be tuned without changing the algorithm.

    Use validation_mode="strict" if you want every candidate checked with eigvalsh.
    Use validation_mode="fast" for the normal Monte Carlo runs.
    """

    def __init__(
        self,
        y_sorted,
        z_sorted,
        pik_sorted,
        var_srs_y,
        var_srs_z,
        M,
        n,
        case_name="",
        objective: str = "eff_z",
        enforce_cadsd_order: bool = True,
        random_state: Optional[int] = None,
        validation_mode: str = "fast",
        eigen_check_interval: int = 0,
        initial_strict_checks: int = 2,
    ):
        self.rng = np.random.default_rng(random_state)

        self.input_y = np.asarray(y_sorted, dtype=float).ravel()
        self.input_z = np.asarray(z_sorted, dtype=float).ravel()
        self.input_pi = np.asarray(pik_sorted, dtype=float).ravel()

        if not (len(self.input_y) == len(self.input_z) == len(self.input_pi)):
            raise ValueError("y_sorted, z_sorted, and pik_sorted must have the same length.")
        if np.any(self.input_pi <= 0):
            raise ValueError("All inclusion probabilities must be positive.")

        self.enforce_cadsd_order = enforce_cadsd_order
        if enforce_cadsd_order:
            self.order = np.argsort(self.input_pi)[::-1]
        else:
            self.order = np.arange(len(self.input_pi))

        self.y_sorted = self.input_y[self.order]
        self.z_sorted = self.input_z[self.order]
        self.pik_sorted = self.input_pi[self.order]

        self.var_srs_y = float(var_srs_y)
        self.var_srs_z = float(var_srs_z)
        self.M = int(M)
        self.n = int(n)
        self.N = len(self.pik_sorted)
        self.case_name = case_name
        self.objective = objective

        if validation_mode not in {"fast", "strict"}:
            raise ValueError("validation_mode must be 'fast' or 'strict'.")
        self.validation_mode = validation_mode
        self.eigen_check_interval = int(eigen_check_interval) if eigen_check_interval else 0
        self.initial_strict_checks = int(initial_strict_checks)

        # Kept for compatibility with downstream sensitivity code.
        self.I_N = np.eye(self.N)
        self.Dpi_inv = np.diag(1.0 / self.pik_sorted)

        # Faster variance vectors: HT variance is (y/pi)' A (y/pi).
        self.y_over_pi = self.y_sorted / self.pik_sorted
        self.z_over_pi = self.z_sorted / self.pik_sorted

        # CaDsd returns K in descending-pi order.
        self.pik_sorted_desc = np.sort(self.input_pi)[::-1]

        self._calculate_optimal()

        # Best solution tracking
        self.global_best_eff_z = 0.0
        self.global_best_eff_y = 0.0
        self.global_best_score = -np.inf
        self.global_best_omega = None
        self.global_best_rho = None

        # History and counters
        self.history = []
        self.history_records = []
        self.scout_history = []
        self.eval_count = 0
        self.valid_count = 0
        self.best_update_count = 0
        self.strict_check_count = 0

    def _variance_pair_from_kernel(self, Kmat: np.ndarray, diag_K: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Fast HT variance calculation for a DSD kernel.

        A = (I-K) * K entrywise, so off-diagonal A_ij = -|K_ij|^2 and
        diagonal A_ii = pi_i(1-pi_i). This avoids two dense diagonal matrix
        multiplications at every candidate evaluation.
        """
        if diag_K is None:
            diag_K = np.real(np.diag(Kmat))

        A = -np.abs(Kmat) ** 2
        np.fill_diagonal(A, diag_K * (1.0 - diag_K))

        var_y = float(np.real(self.y_over_pi @ (A @ self.y_over_pi)))
        var_z = float(np.real(self.z_over_pi @ (A @ self.z_over_pi)))
        return var_y, var_z

    def _calculate_optimal(self):
        """Compute the reference P_pi design efficiency."""
        Base_opt = Ppi(self.pik_sorted)
        Ppi_mat = Base_opt @ Base_opt.T
        diag_K = np.real(np.diag(Ppi_mat))
        var_opt_y, var_opt_z = self._variance_pair_from_kernel(Ppi_mat, diag_K=diag_K)

        self.var_opt_y = var_opt_y
        self.var_opt_z = var_opt_z
        self.eff_y_optimal = self.var_srs_y / var_opt_y if var_opt_y > 0 else np.inf
        self.eff_z_optimal = self.var_srs_z / var_opt_z if var_opt_z > 0 else np.inf

    def _score(self, eff_z: float, eff_y: float) -> float:
        """Objective used by ABC. Default is z-efficiency."""
        if self.objective == "eff_z":
            return eff_z
        if self.objective == "eff_y":
            return eff_y
        if self.objective == "harmonic":
            if eff_z <= 0 or eff_y <= 0:
                return 0.0
            return 2.0 * eff_z * eff_y / (eff_z + eff_y)
        if self.objective == "mean":
            return 0.5 * (eff_z + eff_y)
        raise ValueError("objective must be one of: 'eff_z', 'eff_y', 'harmonic', 'mean'.")

    def _needs_strict_kernel_check(self) -> bool:
        if self.validation_mode == "strict":
            return True
        if self.eval_count <= self.initial_strict_checks:
            return True
        if self.eigen_check_interval and self.eval_count % self.eigen_check_interval == 0:
            return True
        return False

    def _kernel_passes_validation(self, Kmat: np.ndarray, diag_K: np.ndarray) -> bool:
        if not np.all(np.isfinite(diag_K)):
            return False
        if not np.allclose(diag_K, self.pik_sorted_desc, atol=1e-3):
            return False

        # In fast mode, CaDsd's construction is trusted after the diagonal check.
        # Occasional strict checks can still be requested.
        if not self._needs_strict_kernel_check():
            return True

        self.strict_check_count += 1
        try:
            evals = np.linalg.eigvalsh(Kmat)
        except Exception:
            return False

        if not (np.all(evals >= -1e-3) and np.all(evals <= 1.0 + 1e-3)):
            return False
        if not np.isclose(evals.sum(), self.n, atol=1e-3):
            return False
        return True

    def evaluate(self, omega: np.ndarray, rho: np.ndarray) -> Tuple[float, float, bool]:
        """Evaluate a solution and return eff_z, eff_y, valid."""
        self.eval_count += 1

        try:
            K_dict = CaDsd(pi=self.pik_sorted, M=self.M, omega=omega, rho=rho)
            Kmat = K_dict["K"].astype(np.complex128, copy=False)
        except Exception:
            return (0.0, 0.0, False)

        diag_K = np.real(np.diag(Kmat))
        if not self._kernel_passes_validation(Kmat, diag_K):
            return (0.0, 0.0, False)

        var_y, var_z = self._variance_pair_from_kernel(Kmat, diag_K=diag_K)
        if (not np.isfinite(var_y)) or (not np.isfinite(var_z)) or var_y <= 0 or var_z <= 0:
            return (0.0, 0.0, False)

        eff_y = self.var_srs_y / var_y
        eff_z = self.var_srs_z / var_z
        if not (np.isfinite(eff_y) and np.isfinite(eff_z)):
            return (0.0, 0.0, False)

        self.valid_count += 1
        return (eff_z, eff_y, True)

    def _food_from_arrays(self, omega: np.ndarray, rho: np.ndarray, trial: int = 0) -> Optional[dict]:
        eff_z, eff_y, valid = self.evaluate(omega, rho)
        if not valid:
            return None
        score = self._score(eff_z, eff_y)
        food = {
            "omega": omega,
            "rho": rho,
            "eff_z": eff_z,
            "eff_y": eff_y,
            "score": score,
            "trial": trial,
        }
        self._update_global_best(food)
        return food

    def _update_global_best(self, food: dict) -> bool:
        if food["score"] > self.global_best_score:
            self.global_best_score = food["score"]
            self.global_best_eff_z = food["eff_z"]
            self.global_best_eff_y = food["eff_y"]
            self.global_best_omega = food["omega"].copy()
            self.global_best_rho = food["rho"].copy()
            self.best_update_count += 1
            return True
        return False

    def _random_candidate(self) -> Tuple[np.ndarray, np.ndarray]:
        omega = self.rng.random((self.M, self.N))
        rho = self.rng.random((self.M, self.N - 1))
        return omega, rho

    def _candidate_near_best(self, scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        if self.global_best_omega is None or self.global_best_rho is None:
            return self._random_candidate()
        omega = self.global_best_omega + self.rng.normal(0.0, scale, self.global_best_omega.shape)
        rho = self.global_best_rho + self.rng.normal(0.0, scale, self.global_best_rho.shape)
        return np.clip(omega, 0.0, 1.0), np.clip(rho, 0.0, 1.0)

    def initialize_population(self, colony_size: int, verbose: bool = True) -> List[dict]:
        """Initialize food sources."""
        if verbose:
            print(f" Initializing {colony_size} food sources...")

        population = []

        center_omega = 0.5 * np.ones((self.M, self.N))
        center_rho = 0.5 * np.ones((self.M, self.N - 1))
        food = self._food_from_arrays(center_omega, center_rho)
        if food is not None:
            population.append(food)

        max_attempts = max(colony_size * 20, 120)
        count = 0

        while len(population) < colony_size and count < max_attempts:
            omega, rho = self._random_candidate()
            food = self._food_from_arrays(omega, rho)
            if food is not None:
                population.append(food)

            count += 1
            if verbose and count % 100 == 0:
                print(f"   evaluated {count}, valid sources {len(population)}/{colony_size}", end="\r")

        if verbose:
            print(f"\n    Initialized {len(population)} food sources (requested {colony_size})")
            if len(population) > 0:
                print(f"    Best initial: eff_z={self.global_best_eff_z:.4f}, eff_y={self.global_best_eff_y:.4f}")
            else:
                print("    WARNING: no valid solution found.")

        return population

    def _mutate_food(
        self,
        food: dict,
        partner: dict,
        progress: float,
        mode: str = "abc",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a new candidate with adaptive exploration."""
        adaptive_scale = max(0.12, 1.0 - 0.88 * progress)

        if mode == "local":
            return self._candidate_near_best(scale=0.015 + 0.07 * (1.0 - progress))

        phi_omega = self.rng.uniform(-adaptive_scale, adaptive_scale, food["omega"].shape)
        phi_rho = self.rng.uniform(-adaptive_scale, adaptive_scale, food["rho"].shape)

        new_omega = food["omega"] + phi_omega * (food["omega"] - partner["omega"])
        new_rho = food["rho"] + phi_rho * (food["rho"] - partner["rho"])

        if self.global_best_omega is not None and progress > 0.10:
            pull = self.rng.uniform(0.0, 0.30 * progress)
            new_omega = new_omega + pull * (self.global_best_omega - new_omega)
            new_rho = new_rho + pull * (self.global_best_rho - new_rho)

        jitter = 0.006 * adaptive_scale
        new_omega = new_omega + self.rng.normal(0.0, jitter, new_omega.shape)
        new_rho = new_rho + self.rng.normal(0.0, jitter, new_rho.shape)

        return np.clip(new_omega, 0.0, 1.0), np.clip(new_rho, 0.0, 1.0)

    def _greedy_replace(self, old_food: dict, omega: np.ndarray, rho: np.ndarray) -> dict:
        new_food = self._food_from_arrays(omega, rho)
        if new_food is not None and new_food["score"] > old_food["score"]:
            return new_food
        old_copy = old_food.copy()
        old_copy["trial"] = old_food["trial"] + 1
        return old_copy

    def employed_bee_phase(self, population: List[dict], progress: float) -> List[dict]:
        if len(population) == 0:
            return population

        new_population = []
        for i, food in enumerate(population):
            if len(population) > 1:
                k = int(self.rng.integers(0, len(population) - 1))
                if k >= i:
                    k += 1
            else:
                k = i

            new_omega, new_rho = self._mutate_food(food, population[k], progress, mode="abc")
            new_population.append(self._greedy_replace(food, new_omega, new_rho))

        return new_population

    def onlooker_bee_phase(
        self,
        population: List[dict],
        progress: float,
        onlooker_factor: float = 0.60,
    ) -> List[dict]:
        """
        Onlooker phase with tunable intensity.

        onlooker_factor=1.0 is the standard full onlooker phase.
        onlooker_factor=0.5 to 0.7 is much faster and usually enough for the
        large Monte Carlo grid.
        """
        if len(population) == 0 or onlooker_factor <= 0:
            return population

        scores = np.array([food["score"] for food in population], dtype=float)
        scores = scores - np.min(scores) + 1e-12
        if not np.isfinite(scores).all() or scores.sum() <= 0:
            probs = np.ones(len(population)) / len(population)
        else:
            probs = scores / scores.sum()

        new_population = [food.copy() for food in population]
        n_onlookers = max(1, int(round(onlooker_factor * len(population))))

        for _ in range(n_onlookers):
            i = int(self.rng.choice(len(population), p=probs))
            if len(population) > 1:
                k = int(self.rng.integers(0, len(population) - 1))
                if k >= i:
                    k += 1
            else:
                k = i

            mode = "local" if self.rng.random() < 0.10 else "abc"
            new_omega, new_rho = self._mutate_food(new_population[i], population[k], progress, mode=mode)
            new_population[i] = self._greedy_replace(new_population[i], new_omega, new_rho)

        return new_population

    def scout_bee_phase(self, population: List[dict], limit: int, progress: float) -> Tuple[List[dict], int]:
        if len(population) == 0:
            return population, 0

        new_population = []
        n_abandoned = 0

        for food in population:
            if food["trial"] >= limit:
                n_abandoned += 1

                best_new = None
                attempts = 0
                max_attempts = 25

                while attempts < max_attempts:
                    if self.rng.random() < 0.60:
                        omega, rho = self._candidate_near_best(scale=0.03 + 0.08 * (1.0 - progress))
                    else:
                        omega, rho = self._random_candidate()

                    candidate = self._food_from_arrays(omega, rho)
                    if candidate is not None and (best_new is None or candidate["score"] > best_new["score"]):
                        best_new = candidate
                    attempts += 1

                if best_new is not None:
                    best_new["trial"] = 0
                    new_population.append(best_new)
                else:
                    food_copy = food.copy()
                    food_copy["trial"] = 0
                    new_population.append(food_copy)
            else:
                new_population.append(food)

        return new_population, n_abandoned

    def local_search_phase(self, population: List[dict], progress: float, attempts: int = 2) -> List[dict]:
        """Small memetic improvement around the current global best."""
        if len(population) == 0 or self.global_best_omega is None or attempts <= 0:
            return population

        best_candidate = None
        scale = max(0.004, 0.03 * (1.0 - progress))

        for _ in range(attempts):
            omega, rho = self._candidate_near_best(scale=scale)
            candidate = self._food_from_arrays(omega, rho)
            if candidate is not None and (best_candidate is None or candidate["score"] > best_candidate["score"]):
                best_candidate = candidate

        if best_candidate is not None:
            worst_idx = int(np.argmin([food["score"] for food in population]))
            if best_candidate["score"] > population[worst_idx]["score"]:
                population[worst_idx] = best_candidate

        return population

    def _inject_elite(self, population: List[dict]) -> List[dict]:
        """Ensure the current global best is not lost."""
        if len(population) == 0 or self.global_best_omega is None:
            return population

        scores = np.array([food["score"] for food in population])
        if np.max(scores) + 1e-15 < self.global_best_score:
            worst_idx = int(np.argmin(scores))
            population[worst_idx] = {
                "omega": self.global_best_omega.copy(),
                "rho": self.global_best_rho.copy(),
                "eff_z": self.global_best_eff_z,
                "eff_y": self.global_best_eff_y,
                "score": self.global_best_score,
                "trial": 0,
            }

        return population

    def optimize(
        self,
        colony_size: int = 40,
        max_iterations: int = 100,
        limit: int = 25,
        verbose: bool = True,
        progress_interval: Optional[int] = None,
        local_search_interval: int = 25,
        local_search_attempts: int = 2,
        onlooker_factor: float = 0.60,
        early_stopping: bool = True,
        patience: Optional[int] = None,
        min_iterations: Optional[int] = None,
        rel_tol: float = 1e-5,
        time_limit_seconds: Optional[float] = None,
        random_searcher: Optional["RandomSearchAlgorithm"] = None,
        random_evals_per_iteration: Optional[int] = None,
        random_start_evals: Optional[int] = None,
    ) -> dict:
        """Run the ABC algorithm, optionally with a side-by-side random-search baseline."""
        start_time = time.time()

        if progress_interval is None:
            progress_interval = max(1, max_iterations // 20)
        if patience is None:
            patience = max(25, max_iterations // 4)
        if min_iterations is None:
            min_iterations = max(20, max_iterations // 3)

        use_random = random_searcher is not None
        if use_random:
            if random_evals_per_iteration is None:
                random_evals_per_iteration = max(1, int(round(colony_size * (1.0 + onlooker_factor))))
            if random_start_evals is None:
                random_start_evals = colony_size

        if verbose:
            print("=" * 80)
            print(f" ABC ALGORITHM - {self.case_name}")
            print("=" * 80)
            print(" Configuration:")
            print(f"   colony_size          = {colony_size}")
            print(f"   max_iterations       = {max_iterations}")
            print(f"   abandonment limit    = {limit}")
            print(f"   objective            = {self.objective}")
            print(f"   validation mode      = {self.validation_mode}")
            print(f"   onlooker factor      = {onlooker_factor}")
            print(f"   early stopping       = {early_stopping}, patience={patience}")
            print(f"   time limit           = {time_limit_seconds}")
            if use_random:
                print(f"   random search        = ON")
                print(f"   random start evals   = {random_start_evals}")
                print(f"   random evals/iter    = {random_evals_per_iteration}")
            else:
                print(f"   random search        = OFF")
            print(f"   progress interval    = every {progress_interval} iterations")
            print(f"   local search interval= every {local_search_interval} iterations")
            print(" Reference P_pi:")
            print(f"   eff_z = {self.eff_z_optimal:.4f}")
            print(f"   eff_y = {self.eff_y_optimal:.4f}\n")

        population = self.initialize_population(colony_size, verbose)

        if len(population) == 0:
            if verbose:
                print("\nFAILED: could not initialize any valid solution.")
            return self._prepare_results(time.time() - start_time, colony_size, 0)

        if use_random and random_start_evals and random_start_evals > 0:
            random_searcher.step(random_start_evals, include_center_once=True)

        total_abandoned = 0
        start_best = self.global_best_eff_z
        last_improvement_iter = 0
        best_score_for_stopping = self.global_best_score
        stopped_early = False
        stopped_by_time = False
        completed_iterations = 0

        if verbose:
            if use_random:
                header = (
                    f"{'iter':>7} | {'ABC_z':>10} | {'Rand_z':>10} | "
                    f"{'ABC-Pπ':>9} | {'Rand-Pπ':>9} | {'scouts':>6} | "
                    f"{'ABC valid/eval':>14} | {'Rand valid/eval':>15} | {'elapsed':>8}"
                )
            else:
                header = (
                    f"{'iter':>9} | {'best_z':>10} | {'best_y':>10} | "
                    f"{'Δstart':>9} | {'vs Pπ':>9} | {'scouts':>6} | {'valid/eval':>11} | {'elapsed':>8}"
                )
            print(header)
            print("-" * len(header))

        for iteration in range(max_iterations):
            progress = (iteration + 1) / max_iterations

            old_best = self.global_best_score

            population = self.employed_bee_phase(population, progress)
            population = self.onlooker_bee_phase(population, progress, onlooker_factor=onlooker_factor)
            population, n_abandoned = self.scout_bee_phase(population, limit, progress)
            total_abandoned += n_abandoned

            if local_search_interval and (iteration + 1) % local_search_interval == 0:
                population = self.local_search_phase(
                    population,
                    progress,
                    attempts=local_search_attempts,
                )

            population = self._inject_elite(population)

            if use_random and random_evals_per_iteration and random_evals_per_iteration > 0:
                random_searcher.step(random_evals_per_iteration, include_center_once=False)

            completed_iterations = iteration + 1

            if self.global_best_score > old_best * (1.0 + rel_tol):
                last_improvement_iter = iteration + 1
                best_score_for_stopping = self.global_best_score

            self.history.append(self.global_best_eff_z)
            self.scout_history.append(n_abandoned)
            record = {
                "iteration": iteration + 1,
                "best_eff_z": self.global_best_eff_z,
                "best_eff_y": self.global_best_eff_y,
                "best_score": self.global_best_score,
                "n_abandoned": n_abandoned,
                "eval_count": self.eval_count,
                "valid_count": self.valid_count,
                "strict_check_count": self.strict_check_count,
                "elapsed": time.time() - start_time,
            }
            if use_random:
                record.update({
                    "random_best_eff_z": random_searcher.global_best_eff_z,
                    "random_best_eff_y": random_searcher.global_best_eff_y,
                    "random_best_score": random_searcher.global_best_score,
                    "random_eval_count": random_searcher.eval_count,
                    "random_valid_count": random_searcher.valid_count,
                })
            self.history_records.append(record)

            should_print = (
                verbose
                and (
                    iteration == 0
                    or (iteration + 1) % progress_interval == 0
                    or (iteration + 1) == max_iterations
                )
            )

            if should_print:
                delta_start = (
                    100.0 * (self.global_best_eff_z / start_best - 1.0)
                    if start_best > 0 else 0.0
                )
                vs_ppi = (
                    100.0 * (self.global_best_eff_z / self.eff_z_optimal - 1.0)
                    if self.eff_z_optimal > 0 else np.nan
                )
                valid_ratio = f"{self.valid_count}/{self.eval_count}"
                elapsed = time.time() - start_time

                if use_random:
                    rand_vs_ppi = (
                        100.0 * (random_searcher.global_best_eff_z / self.eff_z_optimal - 1.0)
                        if self.eff_z_optimal > 0 and random_searcher.global_best_eff_z > 0 else np.nan
                    )
                    rand_valid_ratio = f"{random_searcher.valid_count}/{random_searcher.eval_count}"
                    print(
                        f"{iteration+1:7d} | {self.global_best_eff_z:10.4f} | "
                        f"{random_searcher.global_best_eff_z:10.4f} | "
                        f"{vs_ppi:+8.2f}% | {rand_vs_ppi:+8.2f}% | {n_abandoned:6d} | "
                        f"{valid_ratio:>14} | {rand_valid_ratio:>15} | {elapsed:7.1f}s"
                    )
                else:
                    print(
                        f"{iteration+1:9d} | {self.global_best_eff_z:10.4f} | "
                        f"{self.global_best_eff_y:10.4f} | {delta_start:+8.2f}% | "
                        f"{vs_ppi:+8.2f}% | {n_abandoned:6d} | {valid_ratio:>11} | "
                        f"{elapsed:7.1f}s"
                    )

            if (
                early_stopping
                and (iteration + 1) >= min_iterations
                and (iteration + 1 - last_improvement_iter) >= patience
            ):
                stopped_early = True
                if verbose:
                    print(
                        f"    Early stop at iter {iteration+1}: "
                        f"no relative improvement > {rel_tol:g} for {patience} iterations."
                    )
                break

            if (
                time_limit_seconds is not None
                and (iteration + 1) >= min_iterations
                and (time.time() - start_time) >= float(time_limit_seconds)
            ):
                stopped_by_time = True
                if verbose:
                    print(f"    Time stop at iter {iteration+1}: reached {time_limit_seconds}s budget.")
                break

        total_time = time.time() - start_time
        results = self._prepare_results(total_time, colony_size, total_abandoned)
        if use_random:
            random_relative_to_ppi = (
                100.0 * (random_searcher.global_best_eff_z / self.eff_z_optimal - 1.0)
                if self.eff_z_optimal > 0 and random_searcher.global_best_eff_z > 0 else -np.inf
            )
            results.update({
                "random_best_eff_z": random_searcher.global_best_eff_z,
                "random_best_eff_y": random_searcher.global_best_eff_y,
                "random_best_score": random_searcher.global_best_score,
                "random_relative_to_ppi_percent": random_relative_to_ppi,
                "random_eval_count": random_searcher.eval_count,
                "random_valid_count": random_searcher.valid_count,
                "random_history": random_searcher.history.copy(),
                "random_history_records": getattr(random_searcher, "random_history_records", []).copy(),
                "random_search_enabled": True,
                "random_evals_per_iteration": random_evals_per_iteration,
                "random_start_evals": random_start_evals,
            })
        else:
            results["random_search_enabled"] = False
        results["stopped_early"] = stopped_early
        results["stopped_by_time"] = stopped_by_time
        results["completed_iterations"] = completed_iterations
        results["strict_check_count"] = self.strict_check_count
        return results

    def _prepare_results(self, total_time: float, colony_size: int, total_abandoned: int) -> dict:
        relative_to_ppi = (
            100.0 * (self.global_best_eff_z / self.eff_z_optimal - 1.0)
            if self.eff_z_optimal > 0 and self.global_best_eff_z > 0 else -np.inf
        )

        gap_percent = (
            100.0 * (self.eff_z_optimal / self.global_best_eff_z - 1.0)
            if self.eff_z_optimal > 0 and self.global_best_eff_z > 0 else np.inf
        )

        success = self.global_best_eff_z > self.eff_z_optimal

        return {
            "case": self.case_name,
            "best_eff_z": self.global_best_eff_z,
            "best_eff_y": self.global_best_eff_y,
            "best_score": self.global_best_score,
            "best_omega": self.global_best_omega,
            "best_rho": self.global_best_rho,
            "optimal_eff_z": self.eff_z_optimal,
            "optimal_eff_y": self.eff_y_optimal,
            "success": success,
            "gap_percent": gap_percent,
            "relative_to_ppi_percent": relative_to_ppi,
            "improvement_percent": max(relative_to_ppi, 0.0),
            "total_time": total_time,
            "colony_size": colony_size,
            "total_abandoned": total_abandoned,
            "eval_count": self.eval_count,
            "valid_count": self.valid_count,
            "history": self.history.copy(),
            "history_records": self.history_records.copy(),
            "scout_history": self.scout_history.copy(),
            "order": self.order.copy(),
            "enforce_cadsd_order": self.enforce_cadsd_order,
            "validation_mode": self.validation_mode,
            "strict_check_count": self.strict_check_count,
        }

    def print_results(self, results: dict):
        """Print formatted final results."""
        print("=" * 80)
        print(" FINAL RESULTS (ABC)")
        print("=" * 80)
        print(f" Case: {results['case']}")
        print("\n Best solution:")
        print(f"   eff_z = {results['best_eff_z']:.6f}")
        print(f"   eff_y = {results['best_eff_y']:.6f}")
        print("\n Reference P_pi:")
        print(f"   eff_z = {results['optimal_eff_z']:.6f}")
        print(f"   eff_y = {results['optimal_eff_y']:.6f}")
        print("\n Comparison:")
        print(f"   ABC vs P_pi = {results['relative_to_ppi_percent']:+.2f}%")
        print("\n Performance:")
        print(f"   time              = {results['total_time']:.2f}s")
        print(f"   colony size       = {results['colony_size']}")
        print(f"   total scouts      = {results['total_abandoned']}")
        print(f"   valid/eval        = {results['valid_count']}/{results['eval_count']}")
        print(f"   strict checks     = {results.get('strict_check_count', 0)}")
        print(f"   stopped early     = {results.get('stopped_early', False)}")
        print(f"   stopped by time   = {results.get('stopped_by_time', False)}")

        if len(results["history"]) > 0 and results["history"][0] > 0:
            conv = 100.0 * (results["history"][-1] / results["history"][0] - 1.0)
            print("\n Convergence:")
            print(f"   start eff_z = {results['history'][0]:.4f}")
            print(f"   final eff_z = {results['history'][-1]:.4f}")
            print(f"   gain        = {conv:+.2f}%")
        print("=" * 80)




class RandomSearchAlgorithm(ABCAlgorithm):
    """
    Pure random-search baseline using exactly the same CaDsd parameterization
    and the same efficiency evaluation as ABC.

    This is not intended to replace ABC. It is a reference curve: if ABC is
    genuinely learning, ABC_z should improve faster than Rand_z under a similar
    evaluation budget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_evaluated = False
        self.random_history_records = []

    def step(self, n_evals: int = 1, include_center_once: bool = False) -> dict:
        """Run n random candidate evaluations and update the random-search best."""
        start_time = time.time()
        n_evals = int(max(0, n_evals))
        done = 0

        if include_center_once and (not self.center_evaluated) and n_evals > 0:
            center_omega = 0.5 * np.ones((self.M, self.N))
            center_rho = 0.5 * np.ones((self.M, self.N - 1))
            self._food_from_arrays(center_omega, center_rho)
            self.center_evaluated = True
            done += 1

        while done < n_evals:
            omega, rho = self._random_candidate()
            self._food_from_arrays(omega, rho)
            done += 1

        self.history.append(self.global_best_eff_z)
        record = {
            "step_evals": n_evals,
            "best_eff_z": self.global_best_eff_z,
            "best_eff_y": self.global_best_eff_y,
            "best_score": self.global_best_score,
            "eval_count": self.eval_count,
            "valid_count": self.valid_count,
            "strict_check_count": self.strict_check_count,
            "elapsed_step": time.time() - start_time,
        }
        self.random_history_records.append(record)
        return record

    def optimize(
        self,
        max_evaluations: int = 1000,
        verbose: bool = True,
        progress_interval: Optional[int] = None,
        include_center_once: bool = True,
    ) -> dict:
        """Standalone random search, useful for quick debugging."""
        start_time = time.time()
        if progress_interval is None:
            progress_interval = max(1, max_evaluations // 20)

        if verbose:
            print("=" * 80)
            print(f" RANDOM SEARCH - {self.case_name}")
            print("=" * 80)
            print(f" Reference P_pi eff_z = {self.eff_z_optimal:.4f}")
            print(f"{'eval':>9} | {'Rand_z':>10} | {'Rand_y':>10} | {'Rand-Pπ':>9} | {'valid/eval':>11} | {'elapsed':>8}")
            print("-" * 76)

        done = 0
        while done < max_evaluations:
            batch = min(progress_interval, max_evaluations - done)
            self.step(batch, include_center_once=(include_center_once and done == 0))
            done += batch
            if verbose:
                rand_vs_ppi = (
                    100.0 * (self.global_best_eff_z / self.eff_z_optimal - 1.0)
                    if self.eff_z_optimal > 0 and self.global_best_eff_z > 0 else np.nan
                )
                print(
                    f"{done:9d} | {self.global_best_eff_z:10.4f} | "
                    f"{self.global_best_eff_y:10.4f} | {rand_vs_ppi:+8.2f}% | "
                    f"{self.valid_count}/{self.eval_count:>5} | {time.time() - start_time:7.1f}s"
                )

        results = self._prepare_results(time.time() - start_time, colony_size=0, total_abandoned=0)
        results["search_type"] = "random"
        results["max_evaluations"] = max_evaluations
        return results


print("Faster ABCAlgorithm and RandomSearchAlgorithm class loaded. Use validation_mode='strict' only for checking/debugging.")


# ============================================================
# Terminal progress patch: positive ratios only
# ============================================================

def _valid_percent(valid, total):
    return 100.0 * valid / total if total and total > 0 else 0.0


def _ratio_to_ref(value, ref):
    """
    Positive efficiency ratio.
    value/ref = 1.00 means equal to Pπ.
    value/ref = 0.95 means 95% of Pπ.
    value/ref = 1.05 means 5% better than Pπ.
    """
    return value / ref if ref and ref > 0 and value > 0 else np.nan


def _abc_optimize_progress_terminal(
    self,
    colony_size=40,
    max_iterations=100,
    limit=25,
    verbose=True,
    progress_interval=None,
    local_search_interval=25,
    local_search_attempts=2,
    onlooker_factor=0.60,
    early_stopping=True,
    patience=None,
    min_iterations=None,
    rel_tol=1e-5,
    time_limit_seconds=None,
    random_searcher=None,
    random_evals_per_iteration=None,
    random_start_evals=None,
):
    start_time = time.time()

    if progress_interval is None:
        progress_interval = max(1, max_iterations // 20)
    if patience is None:
        patience = max(25, max_iterations // 4)
    if min_iterations is None:
        min_iterations = max(20, max_iterations // 3)

    use_random = random_searcher is not None

    if use_random:
        if random_evals_per_iteration is None:
            random_evals_per_iteration = max(1, int(round(colony_size * (1.0 + onlooker_factor))))
        if random_start_evals is None:
            random_start_evals = colony_size

    if verbose:
        print("=" * 80, flush=True)
        print(f" ABC ALGORITHM - {self.case_name}", flush=True)
        print("=" * 80, flush=True)
        print(" Configuration:", flush=True)
        print(f"   colony_size          = {colony_size}", flush=True)
        print(f"   max_iterations       = {max_iterations}", flush=True)
        print(f"   abandonment limit    = {limit}", flush=True)
        print(f"   objective            = {self.objective}", flush=True)
        print(f"   validation mode      = {getattr(self, 'validation_mode', 'unknown')}", flush=True)
        print(f"   onlooker factor      = {onlooker_factor}", flush=True)
        print(f"   early stopping       = {early_stopping}, patience={patience}", flush=True)
        print(f"   time limit           = {time_limit_seconds}", flush=True)
        print(f"   random search        = {'ON' if use_random else 'OFF'}", flush=True)

        if use_random:
            print(f"   random start evals   = {random_start_evals}", flush=True)
            print(f"   random evals/iter    = {random_evals_per_iteration}", flush=True)

        print(f"   progress interval    = every {progress_interval} iterations", flush=True)
        print(f"   local search interval= every {local_search_interval} iterations", flush=True)
        print(" Reference Pπ efficiency versus SRS:", flush=True)
        print(f"   Pπ_z = {self.eff_z_optimal:.2f}", flush=True)
        print(f"   Pπ_y = {self.eff_y_optimal:.2f}\n", flush=True)

    population = self.initialize_population(colony_size, verbose)

    if len(population) == 0:
        if verbose:
            print("\nFAILED: could not initialize any valid solution.", flush=True)
        return self._prepare_results(time.time() - start_time, colony_size, 0)

    if use_random and random_start_evals and random_start_evals > 0:
        random_searcher.step(random_start_evals, include_center_once=True)

    total_abandoned = 0
    last_improvement_iter = 0
    stopped_early = False
    stopped_by_time = False
    completed_iterations = 0

    if verbose:
        if use_random:
            header = (
                f"{'iter':>5} | "
                f"{'ABC_z/Pπ_z':>10} | {'ABC_y/Pπ_y':>10} | "
                f"{'Rand_z/Pπ_z':>11} | {'Rand_y/Pπ_y':>11} | "
                f"{'scout':>5} | {'ABC val%':>8} | {'Rnd val%':>8} | {'elapsed':>8}"
            )
        else:
            header = (
                f"{'iter':>5} | "
                f"{'ABC_z/Pπ_z':>10} | {'ABC_y/Pπ_y':>10} | "
                f"{'scout':>5} | {'ABC val%':>8} | {'elapsed':>8}"
            )

        print(header, flush=True)
        print("-" * len(header), flush=True)

    for iteration in range(max_iterations):
        progress = (iteration + 1) / max_iterations
        old_best = self.global_best_score

        population = self.employed_bee_phase(population, progress)
        population = self.onlooker_bee_phase(
            population,
            progress,
            onlooker_factor=onlooker_factor,
        )
        population, n_abandoned = self.scout_bee_phase(population, limit, progress)
        total_abandoned += n_abandoned

        if local_search_interval and (iteration + 1) % local_search_interval == 0:
            population = self.local_search_phase(
                population,
                progress,
                attempts=local_search_attempts,
            )

        population = self._inject_elite(population)

        if use_random and random_evals_per_iteration and random_evals_per_iteration > 0:
            random_searcher.step(random_evals_per_iteration, include_center_once=False)

        completed_iterations = iteration + 1

        if self.global_best_score > old_best * (1.0 + rel_tol):
            last_improvement_iter = iteration + 1

        record = {
            "iteration": iteration + 1,
            "best_eff_z": self.global_best_eff_z,
            "best_eff_y": self.global_best_eff_y,
            "best_score": self.global_best_score,
            "abc_z_over_ppi_z": _ratio_to_ref(self.global_best_eff_z, self.eff_z_optimal),
            "abc_y_over_ppi_y": _ratio_to_ref(self.global_best_eff_y, self.eff_y_optimal),
            "n_abandoned": n_abandoned,
            "eval_count": self.eval_count,
            "valid_count": self.valid_count,
            "abc_valid_percent": _valid_percent(self.valid_count, self.eval_count),
            "strict_check_count": getattr(self, "strict_check_count", 0),
            "elapsed": time.time() - start_time,
        }

        if use_random:
            record.update({
                "random_best_eff_z": random_searcher.global_best_eff_z,
                "random_best_eff_y": random_searcher.global_best_eff_y,
                "random_best_score": random_searcher.global_best_score,
                "random_z_over_ppi_z": _ratio_to_ref(random_searcher.global_best_eff_z, self.eff_z_optimal),
                "random_y_over_ppi_y": _ratio_to_ref(random_searcher.global_best_eff_y, self.eff_y_optimal),
                "random_eval_count": random_searcher.eval_count,
                "random_valid_count": random_searcher.valid_count,
                "random_valid_percent": _valid_percent(random_searcher.valid_count, random_searcher.eval_count),
            })

        self.history.append(self.global_best_eff_z)
        self.history_records.append(record)
        self.scout_history.append(n_abandoned)

        should_print = (
            verbose
            and (
                iteration == 0
                or (iteration + 1) % progress_interval == 0
                or (iteration + 1) == max_iterations
            )
        )

        if should_print:
            abc_z_ratio = _ratio_to_ref(self.global_best_eff_z, self.eff_z_optimal)
            abc_y_ratio = _ratio_to_ref(self.global_best_eff_y, self.eff_y_optimal)
            abc_valid = _valid_percent(self.valid_count, self.eval_count)
            elapsed = time.time() - start_time

            if use_random:
                rand_z_ratio = _ratio_to_ref(random_searcher.global_best_eff_z, self.eff_z_optimal)
                rand_y_ratio = _ratio_to_ref(random_searcher.global_best_eff_y, self.eff_y_optimal)
                rand_valid = _valid_percent(random_searcher.valid_count, random_searcher.eval_count)

                print(
                    f"{iteration+1:5d} | "
                    f"{abc_z_ratio:10.2f} | {abc_y_ratio:10.2f} | "
                    f"{rand_z_ratio:11.2f} | {rand_y_ratio:11.2f} | "
                    f"{n_abandoned:5d} | {abc_valid:7.2f}% | {rand_valid:7.2f}% | "
                    f"{elapsed:7.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"{iteration+1:5d} | "
                    f"{abc_z_ratio:10.2f} | {abc_y_ratio:10.2f} | "
                    f"{n_abandoned:5d} | {abc_valid:7.2f}% | {elapsed:7.2f}s",
                    flush=True,
                )

        if (
            early_stopping
            and (iteration + 1) >= min_iterations
            and (iteration + 1 - last_improvement_iter) >= patience
        ):
            stopped_early = True
            if verbose:
                print(
                    f"    Early stop at iter {iteration+1}: "
                    f"no relative improvement > {rel_tol:g} for {patience} iterations.",
                    flush=True,
                )
            break

        if (
            time_limit_seconds is not None
            and (iteration + 1) >= min_iterations
            and (time.time() - start_time) >= float(time_limit_seconds)
        ):
            stopped_by_time = True
            if verbose:
                print(f"    Time stop at iter {iteration+1}: reached {time_limit_seconds}s budget.", flush=True)
            break

    total_time = time.time() - start_time
    results = self._prepare_results(total_time, colony_size, total_abandoned)

    results["abc_z_over_Ppi_z"] = _ratio_to_ref(self.global_best_eff_z, self.eff_z_optimal)
    results["abc_y_over_Ppi_y"] = _ratio_to_ref(self.global_best_eff_y, self.eff_y_optimal)
    results["abc_valid_percent"] = _valid_percent(self.valid_count, self.eval_count)

    if use_random:
        results.update({
            "random_best_eff_z": random_searcher.global_best_eff_z,
            "random_best_eff_y": random_searcher.global_best_eff_y,
            "random_best_score": random_searcher.global_best_score,
            "random_z_over_Ppi_z": _ratio_to_ref(random_searcher.global_best_eff_z, self.eff_z_optimal),
            "random_y_over_Ppi_y": _ratio_to_ref(random_searcher.global_best_eff_y, self.eff_y_optimal),
            "random_eval_count": random_searcher.eval_count,
            "random_valid_count": random_searcher.valid_count,
            "random_valid_percent": _valid_percent(random_searcher.valid_count, random_searcher.eval_count),
            "random_history": random_searcher.history.copy(),
            "random_history_records": getattr(random_searcher, "random_history_records", []).copy(),
            "random_search_enabled": True,
            "random_evals_per_iteration": random_evals_per_iteration,
            "random_start_evals": random_start_evals,
        })
    else:
        results["random_search_enabled"] = False

    results["stopped_early"] = stopped_early
    results["stopped_by_time"] = stopped_by_time
    results["completed_iterations"] = completed_iterations
    results["strict_check_count"] = getattr(self, "strict_check_count", 0)

    return results


ABCAlgorithm.optimize = _abc_optimize_progress_terminal


# ============================================================
# USER SETTINGS: edit only this block for terminal runs
# ============================================================

DATA_PATH = "/home/bardia/projects/graphical-sampling/simulations_abc/populations/real/MU284.csv"
OUTPUT_CSV = "abc_random_mu284_terminal.csv"

TEST_CASES = [
    ("ME84", "P75", "High Corr"),
    ("S82",  "P75", "Medium Corr"),
    ("REG",  "P75", "Low Corr"),
]

SAMPLE_SIZES = [5, 10, 20, 40]
Y_VAR = "P85"

ABC_RANDOM_SEED = 12345
RANDOM_SEARCH_SEED = 54321
OBJECTIVE = "eff_z"      # options: "eff_z", "eff_y", "harmonic", "mean"

# How large is each run?
# Random Search tries:
#   RANDOM_START_EVALS + MAX_ITERATIONS * RANDOM_EVALS_PER_ITERATION
# ABC run size is mainly controlled by:
#   COLONY_SIZE, MAX_ITERATIONS, ONLOOKER_FACTOR
COLONY_SIZE = 20
MAX_ITERATIONS = 80
LIMIT = 25
ONLOOKER_FACTOR = 0.55
LOCAL_SEARCH_ATTEMPTS = 1
LOCAL_SEARCH_INTERVAL = 12
PROGRESS_INTERVAL = 10
VALIDATION_MODE = "fast"
TIME_LIMIT_SECONDS = None

EARLY_STOPPING = False
PATIENCE = 50
MIN_ITERATIONS = 20
REL_TOL = 1e-5

RANDOM_START_EVALS = 20
RANDOM_EVALS_PER_ITERATION = 31


# ============================================================
# MAIN TERMINAL RUN
# ============================================================

def run_mu284_abc_random():
    print("=" * 80, flush=True)
    print("ABC + RANDOM SEARCH ON FULL MU284 DATA", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_csv(DATA_PATH)
    N = len(df)

    print("=" * 80, flush=True)
    print("MU284 DATA LOADED", flush=True)
    print("=" * 80, flush=True)
    print(f"path = {DATA_PATH}", flush=True)
    print(f"N = {N}", flush=True)
    print(f"columns = {list(df.columns)}", flush=True)

    for col in ["ME84", "S82", "REG", "P75"]:
        if col in df.columns and Y_VAR in df.columns:
            corr = np.corrcoef(df[Y_VAR].to_numpy(dtype=float), df[col].to_numpy(dtype=float))[0, 1]
            print(f"Corr({Y_VAR}, {col}) = {corr:.3f}", flush=True)

    all_results = []
    abc_run_details = {}

    for z_name, x_name, corr_label in TEST_CASES:
        print(f"\n{'=' * 80}", flush=True)
        print(f"{corr_label}: z = {z_name}", flush=True)
        print(f"{'=' * 80}", flush=True)

        y_raw = df[Y_VAR].to_numpy(dtype=float)
        z_raw = df[z_name].to_numpy(dtype=float)
        x_raw = df[x_name].to_numpy(dtype=float)

        actual_corr = np.corrcoef(y_raw, z_raw)[0, 1]

        print(f"Corr({Y_VAR}, {z_name}) = {actual_corr:.3f}", flush=True)
        print(f"N = {N} (FULL dataset)", flush=True)

        for n in SAMPLE_SIZES:
            print(f"\n  n = {n}:", flush=True)

            pik_raw = inclusionprobabilities(x_raw, n)

            # Sort according to z / pi, as in the original design construction.
            z_over_pi = z_raw / pik_raw
            sort_idx = np.argsort(z_over_pi)

            y_sorted = y_raw[sort_idx]
            z_sorted = z_raw[sort_idx]
            pik_sorted = pik_raw[sort_idx]

            # SRS variance with the same sample size:
            # Var_SRS(total) = N^2 * (1 - n/N) * s^2 / n
            var_srs_y = N**2 * (1.0 - n / N) * np.var(y_raw, ddof=1) / n
            var_srs_z = N**2 * (1.0 - n / N) * np.var(z_raw, ddof=1) / n

            print(
                f"    ABC params: colony={COLONY_SIZE}, iter={MAX_ITERATIONS}, "
                f"limit={LIMIT}, onlookers={ONLOOKER_FACTOR}, "
                f"progress_every={PROGRESS_INTERVAL}, time_limit={TIME_LIMIT_SECONDS}",
                flush=True,
            )

            print(
                f"    Random Search: start_evals={RANDOM_START_EVALS}, "
                f"evals_per_iter={RANDOM_EVALS_PER_ITERATION}",
                flush=True,
            )

            run_key = f"{z_name}_N{N}_n{n}"
            seed_shift = 1000 * n + len(all_results)

            abc = ABCAlgorithm(
                y_sorted=y_sorted,
                z_sorted=z_sorted,
                pik_sorted=pik_sorted,
                var_srs_y=var_srs_y,
                var_srs_z=var_srs_z,
                M=n,
                n=n,
                case_name=run_key,
                objective=OBJECTIVE,
                enforce_cadsd_order=True,
                random_state=ABC_RANDOM_SEED + seed_shift,
                validation_mode=VALIDATION_MODE,
                eigen_check_interval=0,
                initial_strict_checks=2,
            )

            random_search = RandomSearchAlgorithm(
                y_sorted=y_sorted,
                z_sorted=z_sorted,
                pik_sorted=pik_sorted,
                var_srs_y=var_srs_y,
                var_srs_z=var_srs_z,
                M=n,
                n=n,
                case_name=run_key + "_random",
                objective=OBJECTIVE,
                enforce_cadsd_order=True,
                random_state=RANDOM_SEARCH_SEED + seed_shift,
                validation_mode=VALIDATION_MODE,
                eigen_check_interval=0,
                initial_strict_checks=2,
            )

            res = abc.optimize(
                colony_size=COLONY_SIZE,
                max_iterations=MAX_ITERATIONS,
                limit=LIMIT,
                verbose=True,
                progress_interval=PROGRESS_INTERVAL,
                local_search_interval=LOCAL_SEARCH_INTERVAL,
                local_search_attempts=LOCAL_SEARCH_ATTEMPTS,
                onlooker_factor=ONLOOKER_FACTOR,
                early_stopping=EARLY_STOPPING,
                patience=PATIENCE,
                min_iterations=MIN_ITERATIONS,
                rel_tol=REL_TOL,
                time_limit_seconds=TIME_LIMIT_SECONDS,
                random_searcher=random_search,
                random_start_evals=RANDOM_START_EVALS,
                random_evals_per_iteration=RANDOM_EVALS_PER_ITERATION,
            )

            abc_z_ratio = _ratio_to_ref(res["best_eff_z"], res["optimal_eff_z"])
            abc_y_ratio = _ratio_to_ref(res["best_eff_y"], res["optimal_eff_y"])
            rand_z_ratio = _ratio_to_ref(res["random_best_eff_z"], res["optimal_eff_z"])
            rand_y_ratio = _ratio_to_ref(res["random_best_eff_y"], res["optimal_eff_y"])

            abc_valid_pct = _valid_percent(res["valid_count"], res["eval_count"])
            rnd_valid_pct = _valid_percent(res["random_valid_count"], res["random_eval_count"])

            print("\n    Results:", flush=True)
            print(f"      Pπ_z efficiency vs SRS      = {res['optimal_eff_z']:.2f}", flush=True)
            print(f"      Pπ_y efficiency vs SRS      = {res['optimal_eff_y']:.2f}", flush=True)
            print(f"      ABC_z / Pπ_z                = {abc_z_ratio:.2f}", flush=True)
            print(f"      ABC_y / Pπ_y                = {abc_y_ratio:.2f}", flush=True)
            print(f"      Random_z / Pπ_z             = {rand_z_ratio:.2f}", flush=True)
            print(f"      Random_y / Pπ_y             = {rand_y_ratio:.2f}", flush=True)
            print(f"      ABC valid/try               = {abc_valid_pct:.2f}%", flush=True)
            print(f"      Random valid/try            = {rnd_valid_pct:.2f}%", flush=True)
            print(f"      Time                        = {res['total_time']:.2f}s", flush=True)

            all_results.append({
                "z": z_name,
                "corr": actual_corr,
                "n": n,
                "N": N,
                "Ppi_z_eff_vs_SRS": res["optimal_eff_z"],
                "Ppi_y_eff_vs_SRS": res["optimal_eff_y"],
                "ABC_z": res["best_eff_z"],
                "ABC_y": res["best_eff_y"],
                "Random_z": res["random_best_eff_z"],
                "Random_y": res["random_best_eff_y"],
                "ABC_z_over_Ppi_z": abc_z_ratio,
                "ABC_y_over_Ppi_y": abc_y_ratio,
                "Random_z_over_Ppi_z": rand_z_ratio,
                "Random_y_over_Ppi_y": rand_y_ratio,
                "ABC_valid_percent": abc_valid_pct,
                "Random_valid_percent": rnd_valid_pct,
                "time_seconds": res["total_time"],
                "completed_iterations": res.get("completed_iterations", MAX_ITERATIONS),
                "stopped_early": res.get("stopped_early", False),
                "stopped_by_time": res.get("stopped_by_time", False),
                "abc_eval_count": res["eval_count"],
                "random_eval_count": res["random_eval_count"],
            })

            abc_run_details[run_key] = {
                "abc": abc,
                "random_search": random_search,
                "result": res,
                "sort_idx": sort_idx,
                "z_name": z_name,
                "x_name": x_name,
                "n": n,
                "N": N,
            }

            # Save after every finished case, so results survive interruption.
            pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
            print(f"    Partial results saved to: {OUTPUT_CSV}", flush=True)

    print(f"\n\n{'=' * 80}", flush=True)
    print("FINAL SUMMARY TABLE", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    df_res = pd.DataFrame(all_results)

    summary_cols = [
        "z", "corr", "n", "N",
        "Ppi_z_eff_vs_SRS", "Ppi_y_eff_vs_SRS",
        "ABC_z_over_Ppi_z", "ABC_y_over_Ppi_y",
        "Random_z_over_Ppi_z", "Random_y_over_Ppi_y",
        "ABC_valid_percent", "Random_valid_percent",
        "time_seconds", "completed_iterations",
    ]

    df_print = df_res[summary_cols].copy()
    num_cols = df_print.select_dtypes(include=[np.number]).columns
    df_print[num_cols] = df_print[num_cols].round(2)

    print(df_print.to_string(index=False), flush=True)

    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}", flush=True)
    return df_res, abc_run_details


if __name__ == "__main__":
    run_mu284_abc_random()
