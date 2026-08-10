"""Whole pipeline: surveys in, a YOLO dataset out.

Split is BY SURVEY, never by tile. Tiles from one survey overlap and share the
same ground; putting some in train and some in val leaks and reports a score you
will not see in the field.
"""
import os
import random

import numpy as np

from . import dsp, render, synth, tiles
from .targets import CLASSES


def random_scene(ntrace, rng, *, density=0.004, classes=CLASSES):
    n = max(1, int(ntrace * density))
    margin = 60
    return [(str(rng.choice(classes)),
             float(rng.uniform(margin, max(margin + 1, ntrace - margin))))
            for _ in range(n)]


def make(out_dir, *, n_surveys=20, ntrace=1024, seed=0, spec=None,
         negative_ratio=0.9, clean_survey_ratio=0.35, clean_channels=14, val_ratio=0.2, view_channel=None,
         background=None, render_opts=None, progress=None):
    """-> summary dict. Writes images/{train,val} and labels/{train,val} + data.yaml.

    clean_survey_ratio -> fraction of surveys generated with nothing buried at all.
                          Real road is mostly empty; a detector that never sees empty
                          ground learns to always find something.
    negative_ratio     -> of the tiles that end up with no label, the fraction kept.
    """
    from .spec import DEFAULT
    spec = spec or DEFAULT
    ropt = render_opts or dict(do_dewow=True, do_bgr=True, agc_window=80)
    rng = np.random.default_rng(seed)

    n_val = max(1, int(n_surveys * val_ratio))
    val_ids = set(rng.choice(n_surveys, n_val, replace=False).tolist())  # SPLIT BY SURVEY

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    stat = dict(surveys=n_surveys, tiles=0, dropped=0, negatives=0,
                per_class={c: 0 for c in CLASSES}, train=0, val=0)

    n_clean = int(round(n_surveys * clean_survey_ratio))
    clean_ids = set(rng.choice(n_surveys, n_clean, replace=False).tolist())

    for sid in range(n_surveys):
        srng = np.random.default_rng(seed * 10_000 + sid)
        # Some surveys are plain road with nothing under it. Without them the detector
        # only ever sees ground that contains something and learns to always find
        # something. Most real road is empty, and the model has to know that.
        scene = [] if sid in clean_ids else random_scene(ntrace, srng)
        cube, boxes = synth.generate(ntrace, scene, seed * 10_000 + sid, spec,
                                     background=background)
        split = "val" if sid in val_ids else "train"

        # Render the long view AT the channel the target sits under. A target is
        # localised across-track, so a random channel would show almost nothing —
        # that is how you end up with a dataset that is 95% empty ground.
        if view_channel is not None:
            channels = [view_channel]
        elif not boxes:
            # Clean survey. Sample many channels — empty ground is cheap to render and
            # we need enough of it that the detector is not surprised by nothing.
            k = max(1, min(spec.nchannel, clean_channels))
            channels = sorted(srng.choice(spec.nchannel, k, replace=False).tolist())
        else:
            channels = sorted({b[1] for b in boxes})
            empty = [c for c in range(spec.nchannel) if c not in channels]
            if empty:
                channels.append(int(srng.choice(empty)))

        for ch in channels:
            sec = dsp.process_section(cube[:, ch, :], **ropt)
            for t0, t1 in tiles.tile_ranges(cube.shape[0]):
                lines, dropped = tiles.labels_for_tile(boxes, t0, t1, spec.nsample,
                                                       CLASSES, channel=ch)
                if dropped:
                    stat["dropped"] += 1
                    continue
                if not lines and srng.random() > negative_ratio:
                    continue                              # keep only some empty ground
                name = f"s{sid:04d}_c{ch:02d}_{t0}-{t1}"
                render.to_image(sec[t0:t1], clip=99.0).save(
                    os.path.join(out_dir, "images", split, name + ".png"))
                with open(os.path.join(out_dir, "labels", split, name + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                stat["tiles"] += 1
                stat[split] += 1
                if not lines:
                    stat["negatives"] += 1
                for l in lines:
                    stat["per_class"][CLASSES[int(l.split()[0])]] += 1
        if progress:
            progress(sid + 1, n_surveys)

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    return stat
