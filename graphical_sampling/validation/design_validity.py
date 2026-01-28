from __future__ import annotations

from typing import Tuple
import numpy as np

from graphical_sampling.design import Design


# ======================================================
# 1) Check that sample probabilities sum to 1
# ======================================================
def check_probability_sum(
    design: Design,
    tol: float = 1e-8,
) -> Tuple[bool, float]:
    """
    Check whether the sum of sample probabilities equals 1
    up to a numerical tolerance.

    Parameters
    ----------
    design : Design
        Sampling design to validate.
    tol : float
        Numerical tolerance for floating-point error.

    Returns
    -------
    ok : bool
        True if |sum(p) - 1| <= tol.
    prob_sum : float
        Sum of sample probabilities.
    """
    prob_sum = sum(sample.probability for sample in design)
    ok = abs(prob_sum - 1.0) <= tol
    return ok, prob_sum


# ======================================================
# 2) Reconstruct FIP from samples and compare
# ======================================================
def check_fip_consistency(
    design: Design,
    original_pi: np.ndarray,
    tol: float = 1e-8,
) -> Tuple[bool, float, np.ndarray]:
    """
    Reconstruct first-order inclusion probabilities (FIP)
    from the samples and compare with original inclusion
    probabilities.

    Parameters
    ----------
    design : Design
        Sampling design.
    original_pi : np.ndarray
        Original inclusion probabilities (pi).
    tol : float
        Numerical tolerance.

    Returns
    -------
    ok : bool
        True if max |pi - reconstructed_pi| <= tol.
    max_diff : float
        Maximum absolute difference.
    reconstructed_pi : np.ndarray
        Reconstructed inclusion probabilities.
    """
    n_units = len(original_pi)
    reconstructed_pi = np.zeros(n_units)

    # Sum probabilities of samples that include each unit
    for sample in design:
        for idx in sample.ids:
            reconstructed_pi[idx] += sample.probability

    diff = np.abs(original_pi - reconstructed_pi)
    max_diff = float(diff.max())
    ok = max_diff <= tol

    return ok, max_diff, reconstructed_pi
