# TSI — Tangent-Secant Similarity Index

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE--TSP-orange)](#citation)
[![Powered by](https://img.shields.io/badge/Powered%20by-Oscilla%20Studio-purple)](#oscilla-studio)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18966516.svg)](https://doi.org/10.5281/zenodo.18966516)

> This repository is the **open academic reference implementation** of TSI — designed for reproducibility and peer review.
> The computational experiments and visualizations in the companion paper were developed using **Oscilla Studio** by **Oscilla Labs**.

A multiscale geometric framework for **structural alignment** and **critical transition detection** in time series.

---

## Overview

Classical similarity metrics (correlation, DTW, Fréchet) measure *amplitude proximity*. TSI measures *behavioral alignment* — the geometric consistency of how two signals evolve across multiple time scales simultaneously.

| Property | TSI | DDTW | CORT | DTW |
|---|---|---|---|---|
| Multiscale | ✓ | ✗ | ✗ | ✗ |
| O(n) complexity | ✓ | ✗ | ✓ | ✗ |
| Single-signal mode | ✓ | ✗ | ✗ | ✗ |
| Kernel (Mercer) | ✓ | ✗ | ✗ | ✗ |
| Noise variance | σ²/8k² | — | — | — |

**Key theoretical result:** the Big Secant achieves exactly **4× lower noise variance** than the Standard Secant — proven analytically and confirmed experimentally:

```
Var(standard secant) = σ² / (2k²)
Var(big secant)      = σ² / (8k²)   →  4× reduction  (k-independent)
```

---

## Metrics at a Glance

| Metric | Input | Purpose |
|---|---|---|
| **TSI** | two signals | geometric similarity score ∈ (0,1] |
| **GSCI** | one signal | multi-scale angular consistency ∈ (0,1] |
| **SGM** | one signal | curvature proxy `≈ k²\|f''\|` — O(1), no trig |
| **MIA** | one signal | intersection-triangle area — spikes at breaks |
| **FSA** | one signal | focal stability — sensitive to FM transitions |
| **UGSI** | one signal | unified score combining GSCI + MIA + SGM |
| **EventHorizon** | two signals | H(t) — sign change = early warning |

---

## Installation

```bash
git clone https://github.com/Berkye/TSI-framework.git
cd TSI-framework
pip install -r requirements.txt
```

No build step required — pure Python / NumPy.

---

## Quick Start

```python
import numpy as np
from tsi import tsi_series
from tsi.metrics import gsci_series, sgm_series, mia_series, event_horizon_series

# Two signals
t  = np.arange(1000)
f1 = np.sin(2 * np.pi * 0.05 * t)
f2 = np.where(t < 500,
              np.sin(2 * np.pi * 0.05 * t),   # aligned
              np.sin(2 * np.pi * 0.10 * t))    # diverges at t=500

# Cross-signal similarity
tsi = tsi_series(f1, f2, k=5)
print(f"TSI pre-shift : {np.nanmean(tsi[:500]):.3f}")   # high
print(f"TSI post-shift: {np.nanmean(tsi[500:]):.3f}")   # lower

# Event Horizon — early warning
H = event_horizon_series(f1, f2, k=5)
# H < 0 indicates signals have already diverged

# Single-signal analysis
signal = f2.copy()
gsci = gsci_series(signal, k=5)   # drops at regime change
sgm  = sgm_series(signal,  k=5)   # spikes at curvature discontinuity
mia  = mia_series(signal,  k=5)   # spikes at structural break
```

---

## Repository Structure

```
TSI-framework/
├── tsi/
│   ├── __init__.py     # public API
│   ├── core.py         # point-wise metric functions (all equations)
│   └── metrics.py      # vectorised series-level wrappers
├── examples/
│   ├── 01_regime_shift.py       # synthetic regime detection + Event Horizon
│   ├── 02_single_signal.py      # GSCI / SGM / MIA on one signal
│   ├── 03_eeg_intersession.py   # SEED-IV inter-session consistency
│   └── 04_thermodynamic.py      # water phase transition at 100 °C
├── tests/
│   └── test_tsi.py              # pytest unit tests
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Examples

### 1. Regime Shift Detection
```bash
python examples/01_regime_shift.py
```
Detects a frequency transition at t=500 via TSI drop and Event Horizon sign change.

### 2. Single-Signal Analysis
```bash
python examples/02_single_signal.py
```
GSCI / SGM / MIA jointly locate a structural break — no manual thresholds.

### 3. EEG Inter-Session Consistency
```bash
python examples/03_eeg_intersession.py
```
Compares TSI vs. Pearson for 15 SEED-IV subjects across sessions 1–9 months apart.
Requires the SEED-IV dataset; a synthetic stand-in runs automatically otherwise.

### 4. Thermodynamic Phase Transition
```bash
python examples/04_thermodynamic.py
```
All three metrics detect water boiling at 100 °C without thresholds.

---

## Running Tests

```bash
pytest tests/ -v
```

Key assertions verified:
- Identical signals → TSI ≈ 1
- Linear signal → SGM ≈ 0, GSCI ≈ 1, MIA ≈ 0
- **4× variance reduction** (5000-trial Monte Carlo): ratio ∈ [3.5, 4.5]
- SGM ≈ k²|f''(t)| for smooth functions (Lemma 1)
- Boundary indices return NaN

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `k` | 5 | Scale window |
| `alpha` | 1.0 | Angular sensitivity in TSI and GSCI |
| `beta` | 1.0 | Slope-divergence sensitivity |
| `lambda1` | 1.0 | MIA penalty weight in UGSI |
| `lambda2` | 1.0 | SGM penalty weight in UGSI |

**Normalisation:** Z-score or min-max normalise signals before computing TSI when `beta > 0`.

> **Automatic parameter optimization** — including adaptive k selection, domain-specific presets, and real-time tuning — is available in **[Oscilla Studio](#oscilla-studio)**.

---

## Mathematical Background

The full derivation, proofs, and experimental validation are in the companion paper. Core equations:

**TSI:**
$$\text{TSI}(t) = e^{-\alpha\theta} \cdot e^{-\beta D}$$

**Variance reduction (Theorem):**
$$\text{Var}(m') = \frac{\sigma^2}{2k^2}, \quad \text{Var}(m'') = \frac{\sigma^2}{8k^2}$$

**SGM curvature (Lemma 1):**
$$\text{SGM}(t) = k^2 |f''(t)| + O(k^4)$$

---

## Oscilla Studio

This repository gives you the **full mathematical core** of TSI — open and free forever.

For production and research use cases that require more, **Oscilla Studio** by **Oscilla Labs** provides:

- 🎛️ **Automatic parameter optimization** — adaptive k selection and sensitivity tuning
- ⚡ **Real-time streaming** — live data feeds (EEG devices, IoT sensors)
- 📊 **Visual analysis environment** — no-code pipeline builder
- 🧠 **Domain-specific presets** — EEG, biomedical, and signal intelligence workflows
- 📁 **Export & reporting** — publication-ready figures and structured reports

**[→ Oscilla Labs](https://github.com/Berkye)**

---

## Citation

```bibtex
@article{yucetin2026tsi,
  author  = {Y{\"u}cetin, Berk},
  title   = {{Tangent--Secant Similarity Index (TSI)}: A Multiscale Geometric
             Framework for Structural Alignment and Critical Transition
             Detection in Time Series},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2026},
  note    = {arXiv preprint}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Berk Yücetin · Department of Mathematics, Hacettepe University  
berkyucetin25@hacettepe.edu.tr
