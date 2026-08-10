"""Cut a section into tiles and write YOLO labels.

Two decisions here that cost real accuracy if you get them wrong.

1. Tiles OVERLAP. An object sitting on a cut would otherwise be halved in both
   tiles and learned as two small things.

2. A label survives a cut based on ABSOLUTE WIDTH, not the fraction that remains.
   Ratio is wrong: 40% of a 9-trace cavity is 4 traces (unreadable), 40% of a
   100-trace manhole is 40 traces (obviously a manhole). A hyperbola needs a
   minimum width to look like a hyperbola, and that minimum does not scale with
   the object. Tiles whose label was dropped are dropped too — the pixels are
   still there, and teaching "this is background" would inject false negatives.
"""
TILE = 512
STRIDE = 384          # 128 of overlap
MIN_VISIBLE_TRACES = 24


def tile_ranges(ntrace, tile=TILE, stride=STRIDE):
    if ntrace <= tile:
        return [(0, ntrace)]
    out, s = [], 0
    while s + tile <= ntrace:
        out.append((s, s + tile))
        s += stride
    if out[-1][1] < ntrace:
        out.append((ntrace - tile, ntrace))
    return out


def labels_for_tile(boxes, t0, t1, nsample, classes,
                    min_visible=MIN_VISIBLE_TRACES, channel=None):
    """-> (yolo_lines, dropped) ; dropped=True means throw the tile away."""
    lines, dropped = [], False
    w = t1 - t0
    for cls, ch, x0, y0, x1, y1 in boxes:
        if channel is not None and ch != channel:
            continue
        a, b = max(x0, t0), min(x1, t1)
        if b <= a:
            continue
        # Only objects the tile boundary actually CUT can be fragments. An object
        # that is simply narrow — a pipe is one trace wide — is not a fragment and
        # must be kept, or the class disappears from the training set entirely.
        was_cut = (x0 < t0) or (x1 > t1)
        if was_cut and (b - a) < min_visible:
            dropped = True          # the neighbouring tile has it whole
            continue
        cx, cy = ((a + b) / 2 - t0) / w, ((y0 + y1) / 2) / nsample
        bw, bh = (b - a) / w, (y1 - y0) / nsample
        if not (0 < cx < 1 and 0 < cy < 1):
            continue
        lines.append(f"{classes.index(cls)} {cx:.6f} {cy:.6f} "
                     f"{min(bw,1):.6f} {min(bh,1):.6f}")
    return lines, dropped
