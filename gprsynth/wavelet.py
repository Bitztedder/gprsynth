"""Source wavelet.

Default is a Ricker (Mexican-hat) wavelet — the textbook GPR source model. If you
have a real survey you can estimate the wavelet from it and pass your own array;
nothing else changes. We ship the parametric one so this package carries no data
derived from anyone's survey.
"""
import numpy as np


def ricker(peak_freq_hz: float = 400e6, dt_ns: float = 0.150, length: int = 41) -> np.ndarray:
    """Normalised Ricker wavelet sampled at dt_ns, peak scaled to 1."""
    dt_s = dt_ns * 1e-9
    t = (np.arange(length) - length // 2) * dt_s
    a = (np.pi * peak_freq_hz * t) ** 2
    w = (1 - 2 * a) * np.exp(-a)
    peak = np.abs(w).max() or 1.0
    return (w / peak).astype(np.float64)


def peak_index(w: np.ndarray) -> int:
    return int(np.argmax(np.abs(w)))
