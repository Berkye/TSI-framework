"""
examples/04_thermodynamic.py
-----------------------------
Phase-transition detection on the water boiling curve (Sec. XIX-C).

All three single-signal metrics (GSCI, SGM, MIA) jointly identify
the 100 °C transition without any manual threshold.

Run
---
    python examples/04_thermodynamic.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsi.metrics import gsci_series, sgm_series, mia_series

# ── Thermodynamic curve ───────────────────────────────────────────────────────
N    = 500
temp = np.linspace(20, 150, N)          # °C
c_l  = 4186.0                           # J/(kg·K)  liquid water
c_v  = 2010.0                           # J/(kg·K)  water vapour
L    = 2_260_000.0                      # J/kg       latent heat

heat = np.where(
    temp < 100,
    c_l * (temp - 20),
    c_l * 80 + L + c_v * (temp - 100)
)

np.random.seed(1)
heat_noisy = heat + np.random.normal(0, heat.max() * 0.005, N)

k = 5

# ── Metrics ───────────────────────────────────────────────────────────────────
gsci = gsci_series(heat_noisy, k=k)
sgm  = sgm_series(heat_noisy,  k=k)
mia  = mia_series(heat_noisy,  k=k)

trans_idx = np.argmin(np.abs(temp - 100))

print(f"Phase transition at index {trans_idx}  (T = 100 °C)")
print(f"GSCI at transition : {gsci[trans_idx]:.5f}  "
      f"(pre-mean {np.nanmean(gsci[:trans_idx-5]):.3f})")
print(f"SGM  spike ratio   : {sgm[trans_idx] / (np.nanmean(sgm[:trans_idx-5])+1e-10):.1f}×")
print(f"MIA  spike ratio   : {mia[trans_idx] / (np.nanmean(mia[:trans_idx-5])+1e-10):.1f}×")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

axes[0, 0].plot(temp, heat_noisy / 1e6, color="#1f4e79", lw=0.8, alpha=0.6)
axes[0, 0].plot(temp, heat / 1e6, color="#1f4e79", lw=2.0)
axes[0, 0].axvline(100, color="red", lw=1.5, ls="--", label="100 °C")
axes[0, 0].set_xlabel("Temperature (°C)"); axes[0, 0].set_ylabel("Heat (MJ/kg)")
axes[0, 0].set_title("(a) Heat–Temperature Curve"); axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(temp, gsci, color="#375623", lw=1.5)
axes[0, 1].axvline(100, color="red", lw=1.5, ls="--")
axes[0, 1].set_xlabel("Temperature (°C)"); axes[0, 1].set_ylabel("GSCI")
axes[0, 1].set_title("(b) GSCI drop at transition"); axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(temp, sgm, color="#c55a11", lw=1.5)
axes[1, 0].axvline(100, color="red", lw=1.5, ls="--")
axes[1, 0].set_xlabel("Temperature (°C)")
axes[1, 0].set_ylabel(r"SGM = $k^2|f''|$")
axes[1, 0].set_title("(c) SGM curvature spike"); axes[1, 0].grid(alpha=0.3)

mia_p = np.clip(mia, 0, np.nanpercentile(mia[mia > 0], 99) if np.any(mia > 0) else 1)
axes[1, 1].plot(temp, mia_p, color="#2e75b6", lw=1.2)
axes[1, 1].axvline(100, color="red", lw=1.5, ls="--")
axes[1, 1].set_xlabel("Temperature (°C)"); axes[1, 1].set_ylabel("MIA")
axes[1, 1].set_title("(d) MIA structural break"); axes[1, 1].grid(alpha=0.3)

plt.suptitle("Thermodynamic Phase Transition: H₂O at 100 °C", y=1.01)
plt.tight_layout()
plt.savefig("thermodynamic_output.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → thermodynamic_output.png")
plt.show()
