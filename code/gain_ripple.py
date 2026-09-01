"""Analytic gain ripple from Howe et al., arXiv:2507.07706, Eq. (S.1).

The single-mode estimate assumes a pi round-trip reflection phase:
    dG = 10 log10((1 + G eta |Gamma|²) / (1 - G eta |Gamma|²)).
G and eta are linear power ratios; Gamma is an amplitude reflection coefficient.
The estimate returns NaN outside its domain, G eta |Gamma|² < 1."""

from __future__ import annotations

import numpy as np


def db_to_linear_power(gain_db: np.ndarray) -> np.ndarray:
    """Linear power ratio corresponding to a gain in dB."""
    return 10 ** (np.asarray(gain_db) / 10.0)


def gain_ripple_db(
    gain_linear: np.ndarray, transmittivity: float, reflection_magnitude: np.ndarray
) -> np.ndarray:
    """Peak-to-peak ripple in dB from Eq. (S.1).

    Inputs are linear power gain, power transmittivity, and amplitude reflection.
    Broadcast-compatible arrays are supported; G eta |Gamma|² >= 1 returns NaN."""
    reflection_term = gain_linear * transmittivity * np.asarray(reflection_magnitude) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ripple_db = 10 * np.log10((1 + reflection_term) / (1 - reflection_term))
    return np.where(reflection_term < 1, ripple_db, np.nan)
