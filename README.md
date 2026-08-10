# gprsynth

Labelled training data for ground-penetrating radar, generated instead of annotated.

Detecting voids under a road from GPR is a shape-finding task: a buried object shows
up as a hyperbola, and a human reads thousands of traces looking for them. Training a
detector to do it needs labelled radargrams, and there are almost none — labelling
requires an expert, and the expert is the bottleneck you were trying to remove.

This package sidesteps that. It places objects into a radargram itself, so it already
knows where they are. **The label is a by-product of generation, not a separate job.**

```python
from gprsynth import dataset

stat = dataset.make("out", n_surveys=200, ntrace=2048, seed=0)
# -> images/{train,val}, labels/{train,val} (YOLO format), data.yaml
```

Tiles land in subfolders of `SURVEYS_PER_SHARD` surveys each rather than one flat
directory — tens of thousands of files in a single folder is past what most git hosts
accept, and past what an ordinary `ls` handles gracefully. Sharding by survey rather
than by running count keeps every tile of one survey together, which is the unit this
package splits and reasons about everywhere else. Ultralytics globs recursively and
mirrors the subpath from `images/` to `labels/`, so nothing downstream changes.

No survey data needed. The background is synthesised.

---

## What it models

A point scatterer at depth *d* is seen at two-way range `sqrt(x² + d²)` as the antenna
passes, which draws a hyperbola. Amplitude falls with geometric spreading and
absorption. Five classes are shipped:

| class | polarity | why |
|---|---|---|
| `cavity` | **−1** | air is slower than soil, so a void inverts the reflection |
| `pipe` | +1 | a line source — one scatterer, no cross-track extent |
| `manhole` | +1 | shallow, wide, strong; the rim rings louder than the middle |
| `box` | +1 | buried structure |
| `patch` | +1 | a resurfaced cut — very shallow, flat, no thickness echo |

The array is modelled across-track too. Each channel sees the target at its true
slant range, so the hyperbola sits **deeper** the further the channel is from the
object — that arrival-time shift, not amplitude, is what recovers cross-track
position. Antenna directivity attenuates off-nadir returns on top of that.

A target is labelled in a channel only if, **after display processing**, its peak
clears the clutter standard deviation by a set factor. The check measures the
rendered section rather than predicting from raw amplitude: background removal and
AGC lift weak returns, so a target that looks negligible in the raw cube can be
obvious to the detector. Getting this backwards leaves visible hyperbolas
unlabelled — injected false negatives.

## Honest status

- **`cavity` and `pipe` are solid.** Geometry and polarity match what a void and a
  pipe actually do.
- **`manhole`, `box`, `patch` are plausible approximations that have not been
  validated** against labelled real examples. Use them knowing that.
- The synthetic background is **not** a soil simulation. It reproduces the three
  things a detector keys on — correlated clutter, layering, and the direct wave —
  and nothing else.
- No real survey has been compared against synthetic output in this repository.
  If you have real data, measure the gap before you trust a model trained here.

## Two decisions that cost accuracy if you get them wrong

**Split by survey, never by tile.** Tiles from one survey overlap and share ground.
Splitting by tile leaks, and reports a score the field will not reproduce.

**A cut label survives on absolute width, not fraction.** Forty percent of a 9-trace
cavity is 4 traces — unreadable. Forty percent of a 100-trace manhole is obviously a
manhole. A hyperbola needs a minimum width to look like one, and that minimum does not
scale with the object. Tiles whose label was dropped are dropped too: the pixels are
still there, and teaching "this is background" would inject false negatives.

**One render path.** `dsp.process_section` is what the detector sees, at training time
and at inference time. Two implementations drift, and the model quietly learns a
picture the field never shows it.

**Empty ground is a class too.** A share of surveys is generated with nothing buried
at all (`clean_survey_ratio`). Most real road is empty; a detector that only ever sees
ground containing something learns to always find something. The default mix lands
around 30% empty tiles.

## Bring your own data

```python
dataset.make("out", background=my_cube)     # (ntrace, nchannel, nsample)
```

Reading your vendor's format is your problem — this package takes numpy. That keeps
it vendor-neutral.

Acquisition constants live in `gprsynth/spec.py`. The defaults describe a generic
24-channel road array (0.150 ns sampling, 0.080 m channel pitch, 0.050 m trace pitch,
2 m window) — round numbers for a system of this kind, not any particular cart's
calibration. Replace them with your instrument's and everything downstream follows.

## Verify

```bash
python -m gprsynth.verify out
```

Re-derives the geometry from the files on disk without importing the generator. The
point is not to make you trust the generator — it is to make trusting it unnecessary.
It checks image/label pairing, that the tile range in each filename matches the actual
image width, that class ids and box coordinates are in range, and that no survey
appears in both train and val.

It cannot catch both sides misunderstanding the tiling convention the same way. For
that, look at a preview.

## Install

```bash
pip install numpy pillow
python examples/make_demo.py
```

## Related

Full-waveform simulation with [gprMax](https://www.gprmax.com/) (GPL-3.0) is a stricter
alternative to the analytic scatterer model here — slower, and it needs a soil model.
gprMax is not bundled or required.

## Licence

MIT. See `LICENSE`.
