"""Display processing — the standard GPR chain, written from the textbook.

The detector never sees raw amplitudes; it sees this. So the same function has to
run when you build the training set and when you run inference. Keeping one
implementation is the point — two implementations drift and the model quietly
learns a picture the field never shows it.
"""
import numpy as np


def dewow(sec):
    """Remove the low-frequency drift each trace carries."""
    k = max(3, sec.shape[1] // 20)
    kern = np.ones(k) / k
    base = np.apply_along_axis(lambda c: np.convolve(c, kern, "same"), 1, sec)
    return sec - base


def background_removal(sec):
    """Subtract the mean trace — kills the flat direct wave and horizontal banding."""
    return sec - sec.mean(axis=0, keepdims=True)


def gain(sec, db=0.0):
    return sec * (10.0 ** (db / 20.0)) if db else sec


def agc(sec, window=80):
    """Automatic gain control — equalise energy down the trace."""
    if not window or window < 2:
        return sec
    k = np.ones(window) / window
    env = np.sqrt(np.apply_along_axis(lambda c: np.convolve(c * c, k, "same"), 1, sec))
    return sec / np.maximum(env, env.mean() * 1e-3 or 1e-9)


def process_section(sec, *, do_dewow=True, do_bgr=True, db=0.0, agc_window=0):
    out = np.asarray(sec, np.float64)
    if do_dewow:
        out = dewow(out)
    if do_bgr:
        out = background_removal(out)
    if agc_window:
        out = agc(out, agc_window)
    return gain(out, db)
