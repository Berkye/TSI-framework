"""
examples/01_regime_shift.py
----------------------------
Reproduce the synthetic regime-shift experiment from the paper (Sec. XIX-A).

Signal 1  : constant frequency throughout
Signal 2  : same frequency for t < 500, then doubles at t = 500

Metrics shown:
  - TSI time series → drops after transition
  - Event Horizon H(t) → sign change at transition (early warning)

Run
---
    python examples/01_regime_shift.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsi import tsi_series
from tsi.metrics import event_horizon_series

# ── Parameters ───────────────────────────────────────────────────────────────
np.random.seed(42)
N      = 1000
k      = 5
alpha  = 1.0
beta   = 1.0
sigma  = 0.10          # noise std
t_arr  = np.arange(N)

# ── Signals ───────────────────────────────────────────────────────────────────
f1 = np.sin(2 * np.pi * 0.05 * t_arr)
f2 = np.where(t_arr < 500,
              np.sin(2 * np.pi * 0.05 * t_arr),
              np.sin(2 * np.pi * 0.10 * t_arr))

f1n = f1 + np.random.normal(0, sigma, N)
f2n = f2 + np.random.normal(0, sigma, N)

# ── Compute metrics ───────────────────────────────────────────────────────────
tsi = tsi_series(f1n, f2n, k=k, alpha=alpha, beta=beta)
eh  = event_horizon_series(f1n, f2n, k=k)

# ── Statistics ────────────────────────────────────────────────────────────────
tsi_full  = np.nanmean(tsi)
tsi_pre   = np.nanmean(tsi[10:500])
tsi_post  = np.nanmean(tsi[500:990])

print(f"TSI (full record) : {tsi_full:.3f}")
print(f"TSI (t < 500)     : {tsi_pre:.3f}   [aligned]")
print(f"TSI (t >= 500)    : {tsi_post:.3f}   [diverged]")
print(f"Event Horizon sign change near t=500: "
      f"{np.nansum(np.diff(np.sign(eh[400:600])) != 0)} zero-crossings in [400,600]")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axes[0].plot(t_arr, f1n, alpha=0.4, color="#1f4e79", lw=0.6)
axes[0].plot(t_arr, f1,  color="#1f4e79", lw=1.5, label="$f_1$ (const freq)")
axes[0].plot(t_arr, f2n, alpha=0.4, color="#c00000", lw=0.6)
axes[0].plot(t_arr, f2,  color="#c00000", lw=1.5, label="$f_2$ (freq shift)")
axes[0].axvline(500, color="black", lw=1.2, ls="--")
axes[0].set_ylabel("Amplitude"); axes[0].legend(ncol=2); axes[0].grid(alpha=0.3)
axes[0].set_title("(a) Signals")

axes[1].plot(t_arr, tsi, alpha=0.25, color="#1f4e79", lw=0.6)
# rolling mean for clarity
rm = np.convolve(np.nan_to_num(tsi), np.ones(20)/20, mode='same')
axes[1].plot(t_arr, rm, color="#1f4e79", lw=2.0)
axes[1].axvline(500, color="black", lw=1.2, ls="--")
axes[1].axhline(0.9, color="#c55a11", lw=0.8, ls=":", label="0.9 threshold")
axes[1].set_ylim(0.3, 1.05); axes[1].set_ylabel("TSI"); axes[1].grid(alpha=0.3)
axes[1].set_title(f"(b) TSI  —  pre={tsi_pre:.3f}, post={tsi_post:.3f}")

eh_clipped = np.clip(eh, -300, 300)
axes[2].fill_between(t_arr, 0, np.where(eh_clipped >= 0, eh_clipped, 0),
                     color="#375623", alpha=0.3, label="H>0 (convergent)")
axes[2].fill_between(t_arr, 0, np.where(eh_clipped < 0, eh_clipped, 0),
                     color="#c00000", alpha=0.3, label="H<0 (diverged)")
axes[2].axhline(0, color="black", lw=1.2)
axes[2].axvline(500, color="black", lw=1.2, ls="--")
axes[2].set_ylabel("Event Horizon H(t)"); axes[2].legend(); axes[2].grid(alpha=0.3)
axes[2].set_title("(c) Event Horizon — early warning")
axes[2].set_xlabel("Time $t$")

plt.tight_layout()
plt.savefig("regime_shift_output.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → regime_shift_output.png")
plt.show()
