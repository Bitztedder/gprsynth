"""Put targets into a canvas and hand back the labels for free.

This is the whole idea: because we place the object, we know exactly where it is.
No human ever draws a box. The label is a by-product of generation, not a
separate annotation job.
"""
import math

import numpy as np

from . import canvas as canvas_mod
from . import dsp as dsp_mod
from . import targets as targets_mod
from . import wavelet as wavelet_mod


def _add_wavelet(trace, centre_sample, amp, w, wpk):
    start = int(round(centre_sample)) - wpk
    a, b = max(0, start), min(len(trace), start + len(w))
    if b > a:
        trace[a:b] += amp * w[a - start:b - start]


def _place(sec, x_axis_m, x0_m, depth_m, amp, polarity, spec, w, wpk, decay_m=0.6):
    """One point scatterer -> one hyperbola.

    Range grows as sqrt(x^2 + depth^2) as the antenna passes, so the two-way time
    traces a hyperbola. Amplitude falls with spreading and absorption.
    """
    r = np.hypot(x_axis_m - x0_m, depth_m)
    s = (2 * r / spec.velocity_m_per_ns) / spec.dt_ns
    att = (depth_m / np.maximum(r, depth_m)) ** 2 * np.exp(-(r - depth_m) / decay_m)
    for i in range(len(x_axis_m)):
        if s[i] < spec.nsample + len(w):
            _add_wavelet(sec[i], s[i], polarity * amp * att[i], w, wpk)


def _label_box(x0_tr, depth_m, width_m, extra, spec, wlen):
    """Box round the apex and the first `cut` samples of the limbs.

    NOT the full extent of the visible hyperbola. A limb asymptotes and stays
    faintly visible for as long as the section is wide, so "everything you can
    see" is not a box anyone can draw — for a loud shallow target it would be
    the whole tile. The convention here is apex-plus-a-fixed-slice, which is
    what hyperbola detectors are normally trained against.

    Two consequences to know about. Limbs extend outside the box, so a detector
    trained here learns the apex signature and not the full V. And `cut` is a
    constant, decided ahead of time — unlike the visible/not-visible decision in
    generate(), which is measured off the rendered section. If you want boxes
    that track what survives display processing, this is the function to change.
    """
    s0 = spec.sample_of_depth(depth_m)
    cut = 18  # how many samples of the limbs we consider part of the object
    v, dt = spec.velocity_m_per_ns, spec.dt_ns
    dx = math.sqrt((depth_m + 0.5 * cut * dt * v) ** 2 - depth_m ** 2)
    hw = max(1.0, dx / spec.d_trace_m) + width_m / 2 / spec.d_trace_m
    return (x0_tr - hw, s0 - wlen / 2 - extra, x0_tr + hw, s0 + cut + wlen / 2 + extra)


def generate(ntrace, scene, seed, spec, *, background=None, wavelet=None,
             visible_snr=3.5, beam_exponent=4.0):
    """scene = [(class_name, x0_trace), ...]

    -> (cube float32 [ntrace, nchannel, nsample], boxes[(cls, channel, x0, y0, x1, y1)])

    background=None  -> synthetic canvas (publishable)
    background=cube  -> your own recording
    visible_snr      -> a target is labelled in a channel only if, AFTER display
                        processing, its peak clears this many times the clutter
                        standard deviation. Measured, not predicted.
    """
    rng = np.random.default_rng(seed)
    w = wavelet if wavelet is not None else wavelet_mod.ricker(dt_ns=spec.dt_ns)
    wpk = wavelet_mod.peak_index(w)

    cube = (canvas_mod.synthetic(ntrace, spec, rng) if background is None
            else canvas_mod.from_array(background, ntrace, rng))

    x_axis = np.arange(ntrace) * spec.d_trace_m
    pending = []
    for cls, x0_tr in scene:
        scat, depth, width, extra = targets_mod.sample_target(cls, x0_tr, rng, spec)
        centre_ch = int(rng.integers(0, spec.nchannel))
        for ch in range(spec.nchannel):
            # The target sits under one channel; every other channel sees it at a
            # longer slant range, so it appears deeper and weaker — but it DOES appear.
            offset = abs(ch - centre_ch) * spec.d_channel_m
            slant = math.hypot(depth, offset)
            # Antenna directivity. Without it every channel sees the target almost
            # equally and the plan view becomes a uniform band across the array —
            # the cross-track position is then unrecoverable, which is the whole
            # reason a multi-channel array exists.
            cos_theta = depth / slant
            beam = cos_theta ** beam_exponent
            for x_tr, amp, pol in scat:
                _place(cube[:, ch, :], x_axis, x_tr * spec.d_trace_m, slant,
                       amp * beam, pol, spec, w, wpk)
            pending.append((cls, ch, x0_tr, slant, width, extra))

    # Decide visibility by MEASURING the rendered section, not by predicting from raw
    # amplitude. Display processing (background removal, AGC) lifts weak returns, so a
    # target that looks negligible in the raw cube can be obvious to the detector. Get
    # this backwards and you leave visible hyperbolas unlabelled — injected false
    # negatives, the exact failure the tiling rule elsewhere is careful to avoid.
    boxes = []
    per_channel = {}
    for cls, ch, x0_tr, slant, width, extra in pending:
        if ch not in per_channel:
            per_channel[ch] = dsp_mod.process_section(cube[:, ch, :], do_dewow=True,
                                                      do_bgr=True, agc_window=80)
        sec = per_channel[ch]
        box = _label_box(x0_tr, slant, width, extra, spec, len(w))
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, x1 = max(0, x0), min(cube.shape[0], x1)
        y0, y1 = max(0, y0), min(spec.nsample, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = np.abs(sec[x0:x1, y0:y1])
        floor = np.abs(sec).std() or 1e-9
        if patch.max() / floor >= visible_snr:
            boxes.append((cls, ch) + box)
    return cube.astype(np.float32), boxes
