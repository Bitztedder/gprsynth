"""Buried-object models.

Each class returns a set of point scatterers at a common depth. A point scatterer
becomes a hyperbola in the radargram because the two-way range grows as the antenna
moves past it — that is the whole reason GPR interpretation is a shape-finding task.

Amplitudes are deliberately SUBTLE (a few times the local clutter, not twenty times).
Targets that stand out too much teach the detector a problem that does not exist.

Honest status: `cavity` and `pipe` are solid — the geometry and polarity match what a
void and a pipe actually do. `manhole`, `box`, `patch` are plausible approximations
that have NOT been validated against labelled real examples. Treat them accordingly.
"""
import numpy as np

CLASSES = ["cavity", "pipe", "manhole", "box", "patch"]


def _spread(x0_tr, width_m, d_trace_m, n_min):
    n = max(n_min, int(width_m / d_trace_m))
    return np.linspace(-width_m / 2, width_m / 2, n)


def sample_target(cls, x0_tr, rng, spec):
    """-> (scatterers[(x_trace, amplitude, polarity)], depth_m, width_m, extra_samples)

    polarity  +1 = high-impedance reflector (metal, concrete)
              -1 = void — the air interface flips the reflection
    """
    dtr, v, dt = spec.d_trace_m, spec.velocity_m_per_ns, spec.dt_ns
    samp = lambda thickness_m: int((2 * thickness_m / v) / dt / 2)

    if cls == "pipe":
        # A pipe is a line source: one scatterer, no cross-track extent.
        depth = rng.uniform(0.5, 1.5)
        return [(x0_tr, rng.uniform(300, 650), +1)], depth, 0.0, 0

    if cls == "cavity":
        # A void reflects with INVERTED polarity — air is slower than soil.
        depth, width = rng.uniform(0.4, 1.0), rng.uniform(0.3, 0.7)
        amp = rng.uniform(350, 750)
        return ([(x0_tr + o / dtr, amp, -1) for o in _spread(x0_tr, width, dtr, 2)],
                depth, width, samp(0.10))

    if cls == "manhole":
        # Shallow, wide, strong — and the rim rings louder than the middle.
        depth, width = rng.uniform(0.1, 0.4), rng.uniform(0.6, 1.0)
        amp = rng.uniform(1100, 2200)
        return ([(x0_tr + o / dtr, amp * (1.5 if abs(o) > width * 0.4 else 1.0), +1)
                 for o in _spread(x0_tr, width, dtr, 3)], depth, width, samp(0.08))

    if cls == "box":
        depth, width = rng.uniform(0.5, 1.4), rng.uniform(0.5, 1.2)
        amp = rng.uniform(350, 700)
        return ([(x0_tr + o / dtr, amp, +1) for o in _spread(x0_tr, width, dtr, 3)],
                depth, width, samp(0.15))

    # patch — a resurfaced cut. Very shallow, flat, no thickness echo.
    depth, width = rng.uniform(0.05, 0.25), rng.uniform(0.4, 1.0)
    amp = rng.uniform(450, 900)
    return ([(x0_tr + o / dtr, amp, +1) for o in _spread(x0_tr, width, dtr, 4)],
            depth, width, 0)
