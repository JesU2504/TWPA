"""Bloch-wave extraction for reciprocal HFSS supercell two-ports.

The nonlinear solver needs propagation and interface parameters of the
periodically repeated structure, not the transmission phase and reflection of
a finite 50-ohm-terminated device.  This module converts a measured supercell
S matrix to its branch-continuous forward Bloch eigenmode and can install those
parameters into a ``TWPAnalysis.data`` mapping.

The S21 phase is used only as a branch guide.  Its numerical value is never
used as the Bloch propagation phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

import numpy as np


@dataclass(frozen=True)
class BlochParameters:
    """Forward Bloch parameters per simulated supercell."""

    phase: np.ndarray
    attenuation: np.ndarray
    impedance: np.ndarray
    reflection: np.ndarray
    eigenvalue: np.ndarray
    reciprocity_error: np.ndarray


def s_to_abcd(s: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Convert an array of two-port S matrices to ABCD matrices."""
    values = np.asarray(s, dtype=complex)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("s must have shape (frequency, 2, 2)")
    s11 = values[:, 0, 0]
    s12 = values[:, 0, 1]
    s21 = values[:, 1, 0]
    s22 = values[:, 1, 1]
    if np.any(np.abs(s21) <= np.finfo(float).tiny):
        raise ValueError("Bloch extraction is undefined where S21 is zero")
    result = np.empty_like(values)
    result[:, 0, 0] = ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    result[:, 0, 1] = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    result[:, 1, 0] = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * s21 * z0)
    result[:, 1, 1] = ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)
    return result


def _phase_candidate(principal: float, guide: float) -> float:
    turns = round((guide - principal) / (2 * np.pi))
    return principal + 2 * np.pi * turns


def extract_bloch_parameters(s: np.ndarray, z0: float = 50.0) -> BlochParameters:
    """Extract the forward complex Bloch eigenmode of a supercell.

    The ABCD matrix is normalized to unit determinant at each frequency.  This
    removes only the small reciprocity error of the numerical export and makes
    the two eigenvalues reciprocal.  Of the two eigenvalues, the forward one
    is selected by continuity with the positive delay branch; in a stopband,
    the eigenvalue with magnitude greater than one is selected because ABCD
    matrices transfer fields from port 2 back to port 1.
    """
    values = np.asarray(s, dtype=complex)
    abcd = s_to_abcd(values, z0=z0)
    determinants = abcd[:, 0, 0] * abcd[:, 1, 1] - abcd[:, 0, 1] * abcd[:, 1, 0]
    if np.any(np.abs(determinants) <= np.finfo(float).tiny):
        raise ValueError("Singular ABCD matrix in Bloch extraction")
    reciprocity_error = determinants - 1
    reciprocal_abcd = abcd / np.sqrt(determinants)[:, None, None]
    guide = -np.unwrap(np.angle(values[:, 1, 0]))

    phase = np.empty(len(values), dtype=float)
    eigenvalue = np.empty(len(values), dtype=complex)
    impedance = np.empty(len(values), dtype=complex)
    previous_phase: float | None = None

    for index, matrix in enumerate(reciprocal_abcd):
        candidates = np.linalg.eigvals(matrix)
        scored: list[tuple[float, float, complex]] = []
        target = guide[index] if previous_phase is None else previous_phase
        for value in candidates:
            branch = _phase_candidate(float(np.angle(value)), float(guide[index]))
            # S21 selects the band number. Continuity breaks the degeneracy at
            # a band edge, while |lambda| selects the decaying forward mode in
            # a stopband.
            continuity = abs(branch - target)
            guide_error = abs(branch - guide[index])
            stopband_preference = -max(float(np.log(abs(value))), 0.0)
            scored.append((guide_error + 1e-3 * continuity, stopband_preference, value))
        _, _, selected = min(scored, key=lambda item: (item[0], item[1]))
        selected_phase = _phase_candidate(float(np.angle(selected)), float(guide[index]))
        phase[index] = selected_phase
        eigenvalue[index] = selected
        previous_phase = selected_phase

        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        eigenvector_denominator = selected - a
        threshold = np.finfo(float).eps * (1 + abs(selected) + abs(a))
        if abs(eigenvector_denominator) > threshold:
            impedance[index] = b / eigenvector_denominator
        elif abs(c) > np.finfo(float).tiny:
            impedance[index] = (selected - d) / c
        else:
            impedance[index] = complex(np.inf)

    attenuation = np.maximum(np.log(np.abs(eigenvalue)), 0.0)
    reflection = (impedance - z0) / (impedance + z0)
    nearly_passive = (np.abs(reflection) > 1) & (np.abs(reflection) < 1 + 1e-9)
    reflection[nearly_passive] /= np.abs(reflection[nearly_passive])
    return BlochParameters(
        phase=phase,
        attenuation=attenuation,
        impedance=impedance,
        reflection=reflection,
        eigenvalue=eigenvalue,
        reciprocity_error=reciprocity_error,
    )


def apply_bloch_parameters(
    data: MutableMapping[str, Any],
    sample_frequencies: np.ndarray,
    sample_s: np.ndarray,
    cells_per_supercell: int,
    *,
    z0: float = 50.0,
) -> dict[str, float | int | list[float] | str]:
    """Replace solver parameters with measured Bloch data in the HFSS band.

    ``k`` and ``alpha`` remain expressed per physical base cell, matching
    ``twpasolver``'s spatial coordinate.  ``gammas`` is the reflection between
    the solver reference impedance and the forward Bloch impedance, rather
    than S11 of a particular finite-length device.
    """
    if cells_per_supercell <= 0:
        raise ValueError("cells_per_supercell must be positive")
    frequencies = np.asarray(data["freqs"], dtype=float)
    samples = np.asarray(sample_frequencies, dtype=float)
    if samples.ndim != 1 or len(samples) != len(sample_s):
        raise ValueError("sample frequencies and S matrices must have equal length")
    if np.any(np.diff(samples) <= 0):
        raise ValueError("sample frequencies must be strictly increasing")

    bloch = extract_bloch_parameters(sample_s, z0=z0)
    valid = (frequencies >= samples[0]) & (frequencies <= samples[-1])
    target = frequencies[valid]

    k = np.asarray(data["k"], dtype=float).copy()
    alpha = np.asarray(data["alpha"], dtype=float).copy()
    gammas = np.asarray(data["gammas"], dtype=complex).copy()
    k[valid] = np.interp(target, samples, bloch.phase) / cells_per_supercell
    alpha[valid] = np.interp(target, samples, bloch.attenuation) / cells_per_supercell
    gammas[valid] = np.interp(target, samples, bloch.reflection.real) + 1j * np.interp(
        target, samples, bloch.reflection.imag
    )
    data["k"] = k
    data["alpha"] = alpha
    data["gammas"] = gammas

    return {
        "method": "ABCD eigenvalue Bloch extraction",
        "frequency_range_GHz": [float(samples[0]), float(samples[-1])],
        "cells_per_supercell": int(cells_per_supercell),
        "reference_impedance_ohm": float(z0),
        "maximum_reciprocity_error": float(np.max(np.abs(bloch.reciprocity_error))),
        "maximum_bloch_reflection_magnitude": float(np.max(np.abs(bloch.reflection))),
        "maximum_attenuation_neper_per_supercell": float(np.max(bloch.attenuation)),
    }
