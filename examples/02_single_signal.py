"""
examples/02_single_signal.py
-----------------------------
Single-signal stability analysis using GSCI, SGM, and MIA.

Shows how all three metrics spike / drop simultaneously at a
frequency-transition point, enabling threshold-free detection.

Run
---
    python examples/02_single_signal.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsi.metrics import gsci_series, sgm_series, mia_series

# ── Signal ────────────────────────────────────────────────────────────────────
np.random.seed(0)
N = 1000
t = np.arange(N)
k = 5

sig = np.where(t < 600,
               np.sin(2 * np.pi * 0.03 * t),
               np.sin(2 * np.pi * 0.08 * t))
sig += np.random.normal(0, 0.08, N)

# ── Metrics ───────────────────────────────────────────────────────────────────
gsci = gsci_series(sig, k=k)
sgm  = sgm_series(sig,  k=k)
mia  = mia_series(sig,  k=k)

print(f"GSCI — pre-transition mean : {np.nanmean(gsci[:590]):.3f}")
print(f"GSCI — post-transition mean: {np.nanmean(gsci[610:]):.3f}")
print(f"SGM  — spike at transition : {np.nanmax(sgm[580:640]):.4f}")
print(f"MIA  — spike at transition : {np.nanmax(mia[580:640]):.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, sig, color="#1f4e79", lw=0.8, alpha=0.7)
axes[0].axvline(600, color="red", lw=1.5, ls="--")
axes[0].set_ylabel("Amplitude"); axes[0].grid(alpha=0.3)
axes[0].set_title("(a) Signal  —  frequency shift at t=600")

axes[1].plot(t, gsci, color="#375623", lw=1.5)
axes[1].axvline(600, color="red", lw=1.5, ls="--")
axes[1].set_ylabel("GSCI"); axes[1].grid(alpha=0.3)
axes[1].set_title("(b) GSCI — multi-scale angular consistency")

axes[2].plot(t, sgm, color="#c55a11", lw=1.2)
axes[2].axvline(600, color="red", lw=1.5, ls="--")
axes[2].set_ylabel(r"SGM = $k^2|f''|$"); axes[2].grid(alpha=0.3)
axes[2].set_title("(c) SGM — curvature proxy (O(1), no trig)")

mia_clipped = np.clip(mia, 0, np.nanpercentile(mia[mia > 0], 98) if np.any(mia > 0) else 1)
axes[3].fill_between(t, 0, mia_clipped, color="#2e75b6", alpha=0.4)
axes[3].axvline(600, color="red", lw=1.5, ls="--")
axes[3].set_ylabel("MIA"); axes[3].grid(alpha=0.3)
axes[3].set_title("(d) MIA — intersection triangle area (structural break)")
axes[3].set_xlabel("Time t")

plt.tight_layout()
plt.savefig("single_signal_output.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → single_signal_output.png")
plt.show()
