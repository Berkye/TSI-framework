"""
tsi/metrics.py
--------------
Vectorised series-level wrappers around the point-wise functions
in core.py.  Each function returns a NumPy array of the same
length as the input, with NaN padding at boundaries.
"""

import numpy as np
from .core import TSI, GSCI, SGM, MIA, FSA, UGSI, EventHorizon


def tsi_series(f: np.ndarray, g: np.ndarray, k: int = 5,
               alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """Compute TSI for every valid time point.

    Parameters
    ----------
    f, g  : 1-D arrays (equal length, normalised recommended)
    k     : scale window
    alpha : angular sensitivity
    beta  : slope-divergence sensitivity

    Returns
    -------
    np.ndarray  shape (n,), NaN at boundaries |t| < 2k or |t| > n-2k
    """
    n = min(len(f), len(g))
    out = np.full(n, np.nan)
    for t in range(2 * k, n - 2 * k):
        out[t] = TSI(f, g, t, k, alpha, beta)
    return out


def gsci_series(f: np.ndarray, k: int = 5, alpha: float = 1.0) -> np.ndarray:
    """Compute GSCI for every valid time point."""
    n = len(f)
    out = np.full(n, np.nan)
    for t in range(2 * k + 1, n - 2 * k - 1):
        out[t] = GSCI(f, t, k, alpha)
    return out


def sgm_series(f: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute SGM for every valid time point.

    Note: SGM ≈ k²·|f''(t)|  —  zero for linear trends,
    spikes at curvature discontinuities.
    """
    n = len(f)
    out = np.full(n, np.nan)
    for t in range(2 * k, n - 2 * k):
        out[t] = SGM(f, t, k)
    return out


def mia_series(f: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute MIA for every valid time point."""
    n = len(f)
    out = np.full(n, np.nan)
    for t in range(2 * k + 1, n - 2 * k - 1):
        out[t] = MIA(f, t, k)
    return out


def fsa_series(f: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute FSA for every valid time point."""
    n = len(f)
    out = np.full(n, np.nan)
    for t in range(2 * k, n - 2 * k):
        out[t] = FSA(f, t, k)
    return out


def ugsi_series(f: np.ndarray, k: int = 5, alpha: float = 1.0,
                lambda1: float = 1.0, lambda2: float = 1.0) -> np.ndarray:
    """Compute UGSI for every valid time point.

    sigma_f is computed ONCE here and passed to each UGSI call,
    keeping the total cost O(n) instead of O(n²).
    """
    n = len(f)
    sigma_f = float(np.std(f))   # computed once — O(n) total
    out = np.full(n, np.nan)
    for t in range(2 * k + 1, n - 2 * k - 1):
        out[t] = UGSI(f, t, k, alpha, lambda1, lambda2, sigma_f=sigma_f)
    return out


def event_horizon_series(f: np.ndarray, g: np.ndarray,
                         k: int = 5) -> np.ndarray:
    """Compute Event Horizon H(t) for every valid time point.

    H > 0  →  convergent (will intersect in future)
    H = 0  →  at critical transition
    H < 0  →  diverged (already separated)
    """
    n = min(len(f), len(g))
    out = np.full(n, np.nan)
    for t in range(2 * k, n - 2 * k):
        h = EventHorizon(f, g, t, k)
        out[t] = h if np.isfinite(h) else np.nan
    return out
