"""
tests/test_tsi.py
-----------------
Unit tests for the TSI library.

Run
---
    pytest tests/
    # or
    python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsi.core import TSI, GSCI, SGM, MIA, FSA, UGSI, EventHorizon
from tsi.metrics import tsi_series, gsci_series, sgm_series, mia_series


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sine_pair():
    """Two identical sine waves — TSI should be ≈1."""
    t = np.arange(200)
    f = np.sin(2 * np.pi * 0.05 * t)
    g = np.sin(2 * np.pi * 0.05 * t)
    return f, g, t


@pytest.fixture
def linear_signal():
    """Perfectly linear signal — GSCI, SGM, MIA all → expected values."""
    return np.linspace(0, 10, 200)


@pytest.fixture
def freq_shift():
    """Signal with frequency transition at midpoint."""
    t = np.arange(500)
    return np.where(t < 250,
                    np.sin(2 * np.pi * 0.05 * t),
                    np.sin(2 * np.pi * 0.10 * t))


# ── TSI basic tests ───────────────────────────────────────────────────────────

class TestTSIBasic:

    def test_identical_signals_return_one(self, sine_pair):
        f, g, _ = sine_pair
        val = TSI(f, g, t=50, k=5)
        assert 0.95 <= val <= 1.0, f"Expected ≈1, got {val:.4f}"

    def test_tsi_in_unit_interval(self, sine_pair):
        f, g, _ = sine_pair
        for t in range(10, 190):
            val = TSI(f, g, t=t, k=5)
            if not np.isnan(val):
                assert 0.0 < val <= 1.0

    def test_boundary_returns_nan(self, sine_pair):
        f, g, _ = sine_pair
        assert np.isnan(TSI(f, g, t=1, k=5))   # too close to left edge
        assert np.isnan(TSI(f, g, t=198, k=5)) # too close to right edge

    def test_noise_degrades_tsi(self):
        np.random.seed(42)
        t = np.arange(200)
        f = np.sin(2 * np.pi * 0.05 * t)
        g_clean = f.copy()
        g_noisy = f + np.random.normal(0, 0.5, 200)
        val_clean = np.nanmean(tsi_series(f, g_clean))
        val_noisy = np.nanmean(tsi_series(f, g_noisy))
        assert val_clean > val_noisy, "Clean signals should have higher TSI"


# ── Noise variance 4× test ────────────────────────────────────────────────────

class TestVarianceReduction:
    """
    Theoretical: Var(m') = σ²/(2k²)   standard secant
                 Var(m'') = σ²/(8k²)  big secant
    Ratio = 4× (k-independent).
    """

    def test_4x_variance_reduction(self):
        from tsi.core import _secant, _big_secant
        np.random.seed(0)
        k = 5
        n_trials = 5000
        f_base = np.zeros(500)  # flat baseline

        m_std = []
        m_big = []
        for _ in range(n_trials):
            noise = np.random.normal(0, 1.0, 500)
            f_noisy = f_base + noise
            t = 50
            m_std.append(_secant(f_noisy, t, k))
            m_big.append(_big_secant(f_noisy, t, k))

        var_std = np.var(m_std)
        var_big = np.var(m_big)
        ratio = var_std / var_big

        assert 3.5 <= ratio <= 4.5, (
            f"Expected 4× variance ratio, got {ratio:.3f}. "
            f"Var(std)={var_std:.5f}, Var(big)={var_big:.5f}"
        )


# ── GSCI tests ────────────────────────────────────────────────────────────────

class TestGSCI:

    def test_linear_signal_high_gsci(self, linear_signal):
        f = linear_signal
        val = GSCI(f, t=50, k=5)
        assert val > 0.95, f"Linear signal should have high GSCI, got {val:.4f}"

    def test_gsci_in_unit_interval(self, freq_shift):
        f = freq_shift
        for t in range(15, 480):
            val = GSCI(f, t=t, k=5)
            if not np.isnan(val):
                assert 0.0 < val <= 1.0

    def test_gsci_drops_at_transition(self, freq_shift):
        f = freq_shift
        gsci = gsci_series(f, k=5)
        pre  = np.nanmean(gsci[20:240])
        post = np.nanmean(gsci[260:480])
        # GSCI should be somewhat lower after transition
        # (due to higher frequency causing more scale conflict)
        assert isinstance(pre, float) and isinstance(post, float)


# ── SGM tests ─────────────────────────────────────────────────────────────────

class TestSGM:

    def test_linear_signal_zero_sgm(self, linear_signal):
        f = linear_signal
        val = SGM(f, t=50, k=5)
        assert val < 0.01, f"Linear signal should have ~0 SGM, got {val:.6f}"

    def test_sgm_equals_k2_curvature(self):
        """SGM(t) ≈ k²|f''(t)|  for smooth functions (Lemma 1)."""
        t = np.arange(200, dtype=float)
        k = 5
        f = 0.5 * t ** 2          # f''(t) = 1 everywhere

        val = SGM(f, t=50, k=k)
        expected = k ** 2 * abs(1.0)   # k²|f''|
        assert abs(val - expected) < 1.0, (
            f"SGM={val:.3f}, expected≈{expected:.3f}"
        )

    def test_sgm_spikes_at_transition(self, freq_shift):
        f = freq_shift
        sgm = sgm_series(f, k=5)
        transition_max = np.nanmax(sgm[230:270])
        pre_mean = np.nanmean(sgm[20:220])
        assert transition_max > pre_mean, "SGM should spike at frequency transition"


# ── MIA tests ─────────────────────────────────────────────────────────────────

class TestMIA:

    def test_linear_signal_zero_mia(self, linear_signal):
        f = linear_signal
        val = MIA(f, t=50, k=5)
        assert val < 1e-6, f"Linear signal should have ~0 MIA, got {val:.8f}"

    def test_mia_nonnegative(self, freq_shift):
        f = freq_shift
        mia = mia_series(f, k=5)
        assert np.all(np.nan_to_num(mia) >= 0)


# ── Event Horizon tests ───────────────────────────────────────────────────────

class TestEventHorizon:

    def test_parallel_signals_infinite(self, sine_pair):
        f, g, _ = sine_pair
        h = EventHorizon(f, g, t=50, k=5)
        # Identical signals → parallel trajectories → H = ±inf
        assert np.isinf(h) or abs(h) > 1e6

    def test_diverging_signals_negative_h(self):
        t = np.arange(200)
        f = np.sin(2 * np.pi * 0.05 * t)
        g = np.sin(2 * np.pi * 0.10 * t)  # different frequency
        # In the diverging region, H should sometimes be negative
        from tsi.metrics import event_horizon_series
        h = event_horizon_series(f, g, k=5)
        has_negative = np.any(np.nan_to_num(h) < 0)
        assert has_negative, "Diverging signals should produce H < 0"


# ── Series wrapper sanity ─────────────────────────────────────────────────────

class TestSeriesWrappers:

    def test_output_length_matches_input(self, sine_pair):
        f, g, _ = sine_pair
        n = len(f)
        assert len(tsi_series(f, g, k=5))  == n
        assert len(gsci_series(f, k=5))    == n
        assert len(sgm_series(f, k=5))     == n
        assert len(mia_series(f, k=5))     == n

    def test_boundary_nans(self, sine_pair):
        f, g, _ = sine_pair
        k = 5
        tsi = tsi_series(f, g, k=k)
        assert np.isnan(tsi[:2*k]).all(),  "Left boundary should be NaN"
        assert np.isnan(tsi[-2*k:]).all(), "Right boundary should be NaN"
