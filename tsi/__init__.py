"""
TSI — Tangent-Secant Similarity Index
======================================
A multiscale geometric framework for structural alignment
and critical transition detection in time series.

Author : Berk Yücetin
Affil. : Department of Mathematics, Hacettepe University
Contact: berkyucetin25@hacettepe.edu.tr
License: MIT

This open-source reference implementation accompanies the paper:
  Yücetin, B. (2026). Tangent-Secant Similarity Index (TSI).
  IEEE Transactions on Signal Processing.

The computational experiments in the paper were developed using
Oscilla Studio, a signal analysis platform by Oscilla Labs.
"""

from .core import TSI, GSCI, SGM, MIA, FSA, UGSI, EventHorizon
from .metrics import tsi_series, gsci_series, sgm_series, mia_series

__version__ = "1.0.0"
__all__ = [
    "TSI", "GSCI", "SGM", "MIA", "FSA", "UGSI", "EventHorizon",
    "tsi_series", "gsci_series", "sgm_series", "mia_series",
]
