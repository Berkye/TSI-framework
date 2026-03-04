"""
tsi/core.py
-----------
Point-wise TSI metric computations.

All functions operate on NumPy arrays and follow the
mathematical definitions in the companion paper:

  Yücetin, B. (2026). "Tangent-Secant Similarity Index (TSI):
  A Multiscale Geometric Framework for Structural Alignment and
  Critical Transition Detection in Time Series."
  IEEE Transactions on Signal Processing.

Contact: berkyucetin25@hacettepe.edu.tr

Equations are referenced as (Eq. N) throughout.
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
_EPS = 1e-10   # numerical safety floor
_PI2 = np.pi / 2


# ── Internal helpers ─────────────────────────────────────────────────────────

def _secant(f: np.ndarray, t: int, k: int) -> float:
    """Standard secant slope (central difference quotient).

    m' = [f(t+k) - f(t-k)] / (2k)          (Eq. 3)

    Noise variance: Var(m') = σ² / (2k²)    (Eq. 15)
    """
    return (f[t + k] - f[t - k]) / (2.0 * k)


def _big_secant(f: np.ndarray, t: int, k: int) -> float:
    """Big secant slope (macro-scale central difference).

    m'' = [f(t+2k) - f(t-2k)] / (4k)       (Eq. 5)

    Noise variance: Var(m'') = σ² / (8k²)  (Eq. 16)
    → 4× lower than standard secant (k-independent).
    """
    return (f[t + 2 * k] - f[t - 2 * k]) / (4.0 * k)


def _tangent(f: np.ndarray, t: int) -> float:
    """Central-difference tangent approximation.

    mT ≈ [f(t+1) - f(t-1)] / 2             (Eq. 20)
    """
    return (f[t + 1] - f[t - 1]) / 2.0


def _angle_between(m1: float, m2: float) -> float:
    """Angle between two lines with slopes m1, m2.

    θ = arctan |( m1 - m2 ) / ( 1 + m1·m2 )|   (Eq. 2)

    Special cases:
      1 + m1·m2 = 0  →  θ = π/2  (perpendicular)
      m1 = m2        →  θ = 0    (parallel)
    """
    denom = 1.0 + m1 * m2
    if abs(denom) < _EPS:
        return _PI2
    return abs(np.arctan((m1 - m2) / denom))


def _line_intersect(m1: float, b1: float, m2: float, b2: float):
    """Intersection (x, y) of two lines y = m·x + b.

    Returns (None, None) when lines are parallel.
    """
    denom = m1 - m2
    if abs(denom) < _EPS:
        return None, None
    x = (b2 - b1) / denom
    y = m1 * x + b1
    return x, y


# ── Public point-wise functions ───────────────────────────────────────────────

def TSI(f: np.ndarray, g: np.ndarray, t: int, k: int = 5,
        alpha: float = 1.0, beta: float = 1.0) -> float:
    """Tangent-Secant Similarity Index at time t.

    TSI(t) = exp(-α·θ) · exp(-β·D)          (Eq. 9)

    where
      θ  = angle between secant slopes of f and g
      D  = |m'_f - m'_g|  (slope divergence)

    Parameters
    ----------
    f, g  : 1-D arrays of equal length
    t     : reference time index  (must satisfy 2k ≤ t ≤ n-2k-1)
    k     : scale window  (default 5)
    alpha : angular sensitivity  (default 1.0)
    beta  : slope-divergence sensitivity  (default 1.0)

    Returns
    -------
    float in (0, 1].  Returns NaN if boundary violated.

    Notes
    -----
    Signals should be z-score or min-max normalised before
    applying the beta term to avoid unbounded D values.

    For advanced parameter tuning and real-time analysis,
    see Oscilla Studio by Oscilla Labs.
    """
    n = min(len(f), len(g))
    if t - 2 * k < 0 or t + 2 * k >= n:
        return np.nan

    mf = _secant(f, t, k)
    mg = _secant(g, t, k)

    theta = _angle_between(mf, mg)
    D = abs(mf - mg)

    return np.exp(-alpha * theta) * np.exp(-beta * D)


def GSCI(f: np.ndarray, t: int, k: int = 5, alpha: float = 1.0) -> float:
    """Geometric Self-Consistency Index at time t.

    Measures multi-scale angular agreement within a single signal.

    θ_total = |arctan(mT) - arctan(mS)| + |arctan(mS) - arctan(mB)|  (Eq. 23)
    GSCI(t) = exp(-α · θ_total)                                        (Eq. 24)

    GSCI → 1  :  fractal alignment across scales (stable)
    GSCI → 0  :  scale conflict (regime change imminent)

    Parameters
    ----------
    f     : 1-D array
    t     : reference time index  (must satisfy 2k+1 ≤ t ≤ n-2k-2)
    k     : scale window  (default 5)
    alpha : sensitivity parameter  (default 1.0)
    """
    n = len(f)
    if t - 2 * k - 1 < 0 or t + 2 * k + 1 >= n:
        return np.nan

    mT = _tangent(f, t)
    mS = _secant(f, t, k)
    mB = _big_secant(f, t, k)

    theta_total = (
        abs(np.arctan(mT) - np.arctan(mS)) +
        abs(np.arctan(mS) - np.arctan(mB))
    )
    return np.exp(-alpha * theta_total)


def SGM(f: np.ndarray, t: int, k: int = 5) -> float:
    """Structural Gap Metric at time t.

    SGM(t) = |f(t) - (P_fwd + P_bwd) / 2|   (Eq. 28)

    Taylor expansion gives:  SGM(t) = k²·|f''(t)| + O(k⁴)  (Lemma 1)

    P_fwd = 2·f(t-k) - f(t-2k)   (linear forward extrapolation)
    P_bwd = 2·f(t+k) - f(t+2k)   (linear backward extrapolation)

    Complexity: O(1), no trigonometric operations.
    → Suitable for embedded / real-time applications.
    """
    n = len(f)
    if t - 2 * k < 0 or t + 2 * k >= n:
        return np.nan

    P_fwd = 2.0 * f[t - k] - f[t - 2 * k]
    P_bwd = 2.0 * f[t + k] - f[t + 2 * k]
    return abs(f[t] - (P_fwd + P_bwd) / 2.0)


def MIA(f: np.ndarray, t: int, k: int = 5) -> float:
    """Multiscale Intersection Area at time t.

    Computes the area of the triangle formed by pairwise intersections
    of the tangent line (L_T), secant line (L_S), and big-secant
    line (L_B) at time t.

    MIA(t) = ½ |x₁₂(y₂₃-y₁₃) + x₂₃(y₁₃-y₁₂) + x₁₃(y₁₂-y₂₃)|  (Eq. 25)

    Stable signal   → lines nearly concurrent → MIA ≈ 0
    Regime change   → triangle expands       → MIA spikes
    """
    n = len(f)
    if t - 2 * k - 1 < 0 or t + 2 * k + 1 >= n:
        return np.nan

    mT = _tangent(f, t)
    mS = _secant(f, t, k)
    mB = _big_secant(f, t, k)

    # Line intercepts: y = m·x + b
    bT = f[t] - mT * t
    bS = f[t + k] - mS * (t + k)
    bB = f[t + 2 * k] - mB * (t + 2 * k)

    p12 = _line_intersect(mT, bT, mS, bS)
    p23 = _line_intersect(mS, bS, mB, bB)
    p13 = _line_intersect(mT, bT, mB, bB)

    if any(x is None for x, _ in [p12, p23, p13]):
        return 0.0

    x12, y12 = p12
    x23, y23 = p23
    x13, y13 = p13

    # Shoelace formula
    return 0.5 * abs(
        x12 * (y23 - y13) +
        x23 * (y13 - y12) +
        x13 * (y12 - y23)
    )


def FSA(f: np.ndarray, t: int, k: int = 5) -> float:
    """Focal Stability Area at time t.

    FSA(t) = ½ · Gap(t) · Drift(t)          (Eq. 38)

    where
      Drift = |x_I - t|         (temporal distance to intersection)
      Gap   = |f̄_S  - f̄_B|    (vertical opening at x = t)

    FSA is sensitive to frequency modulation (FM) in periodic signals.
    """
    n = len(f)
    if t - 2 * k < 0 or t + 2 * k >= n:
        return np.nan

    mS = _secant(f, t, k)
    mB = _big_secant(f, t, k)

    fS_bar = f[t]                    # secant passes through (t, f[t])
    fB_bar = f[t + 2 * k] - mB * 2 * k + mB * t  # y at x=t for big secant line

    gap = abs(fS_bar - fB_bar)

    if abs(mS - mB) < _EPS:
        return 0.0

    x_I = t + (fB_bar - fS_bar) / (mS - mB)
    drift = abs(x_I - t)

    return 0.5 * gap * drift


def UGSI(f: np.ndarray, t: int, k: int = 5, alpha: float = 1.0,
         lambda1: float = 1.0, lambda2: float = 1.0,
         sigma_f: float = None) -> float:
    """Unified Geometric Stability Index at time t.

    UGSI(t) = GSCI(t) · exp(-λ₁·MIA(t)/σ²_f - λ₂·SGM(t)/σ_f)   (Eq. 40)

    Combines GSCI (angular consistency), MIA (structural break penalty),
    and SGM (curvature penalty) into a single 0→1 score.

    Parameters
    ----------
    lambda1 : MIA penalty weight  (default 1.0)
    lambda2 : SGM penalty weight  (default 1.0)
    sigma_f : signal std — MUST be pre-computed and passed in for real-time
              use. If None, computed once from f (fine for offline analysis,
              but O(N) overhead per call in streaming mode).

    Performance note
    ----------------
    In streaming / real-time applications, compute sigma_f = np.std(window)
    once per batch and pass it explicitly.  The ugsi_series() wrapper in
    metrics.py does this automatically — use that for series-level work.
    """
    if sigma_f is None:
        sigma_f = np.std(f)
    sigma_f = sigma_f + _EPS

    g = GSCI(f, t, k, alpha)
    m = MIA(f, t, k)
    s = SGM(f, t, k)

    if any(np.isnan(v) for v in [g, m, s]):
        return np.nan

    return g * np.exp(-lambda1 * m / sigma_f**2 - lambda2 * s / sigma_f)


def EventHorizon(f: np.ndarray, g: np.ndarray, t: int, k: int = 5) -> float:
    """Event Horizon at time t.

    H(t) = t_p - t  where  t_p = t + (g(t) - f(t)) / (m'_f - m'_g)  (Eq. 12-13)

    H → +∞  :  stable alignment (parallel trajectories)
    H ≈ 0   :  critical transition (imminent crossing)
    H < 0   :  divergence (signals have already separated)

    Returns np.nan when trajectories are parallel.
    """
    n = min(len(f), len(g))
    if t - 2 * k < 0 or t + 2 * k >= n:
        return np.nan

    mf = _secant(f, t, k)
    mg = _secant(g, t, k)

    denom = mf - mg
    if abs(denom) < _EPS:
        return np.inf

    tp = t + (g[t] - f[t]) / denom
    return tp - t
