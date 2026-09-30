"""Background canvas.

Two ways to get one:

  synthetic(...)  — built from noise. No survey data involved, so anything you
                    generate with it is yours to publish.
  from_array(...) — a numpy cube you supply. The output then carries whatever
                    rights that array carries.

The synthetic canvas is not a physics simulation of soil. It reproduces the three
things a detector actually keys on: correlated noise (the ground is not white),
horizontal layering, and a strong direct-wave band at the top of every trace.
"""
import numpy as np


def _smooth(x, k, axis):
    if k < 2:
        return x
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, kern, "same"), axis, x)


def synthetic(ntrace, spec, rng, *, roughness=6, layers=(3, 7), direct_wave=True):
    """-> cube (ntrace, nchannel, nsample) float64"""
    ns, nch = spec.nsample, spec.nchannel
    cube = rng.standard_normal((ntrace, nch, ns))
    # Ground clutter is correlated in both directions — white noise looks nothing like it.
    cube = _smooth(cube, roughness, axis=2)
    cube = _smooth(cube, max(2, roughness // 2), axis=0)
    cube *= 40.0 / (cube.std() or 1.0)

    # Flat-ish layering: a few interfaces that drift slowly along the line.
    for _ in range(int(rng.integers(*layers))):
        s0 = rng.uniform(ns * 0.15, ns * 0.85)
        drift = _smooth(rng.standard_normal(ntrace), 64, axis=0) * ns * 0.02
        amp = rng.uniform(20, 60) * (1 if rng.random() < 0.5 else -1)
        thick = rng.uniform(2.0, 5.0)
        s = np.arange(ns)[None, :]
        centre = (s0 + drift)[:, None]
        band = amp * np.exp(-((s - centre) ** 2) / (2 * thick ** 2))
        cube += band[:, None, :]

    if direct_wave:
        # Air/ground wave: always present, always at the top, always strongest.
        s = np.arange(ns)
        dw = 900 * np.exp(-((s - 6) ** 2) / (2 * 3.0 ** 2))
        cube += dw[None, None, :]

    # Attenuation with depth — deep returns are quieter, and that matters for gain.
    cube *= np.exp(-np.arange(ns) / (ns * 0.55))[None, None, :]
    return cube


def from_array(cube, ntrace, rng):
    """Tile/flip your own cube to the requested length. Read-only on the input."""
    cube = np.asarray(cube, np.float64)
    if ntrace <= cube.shape[0]:
        s = int(rng.integers(0, cube.shape[0] - ntrace + 1))
        out = cube[s:s + ntrace].copy()
    else:
        reps = [cube, cube[::-1]] * (ntrace // cube.shape[0] // 2 + 2)
        out = np.concatenate(reps, axis=0)[:ntrace].copy()
    return out[::-1].copy() if rng.random() < 0.5 else out
