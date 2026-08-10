"""Independent check of a generated dataset.

Deliberately does NOT import dataset.py. It re-derives the geometry from the files
on disk. The point is not to make you trust the generator — it is to make trusting
it unnecessary. If the generator is wrong, the two disagree and you find out.

What it cannot catch: both sides misunderstanding the tiling convention the same
way. For that you look at a preview with your eyes.
"""
import os
import sys

from PIL import Image


def check(root, classes):
    problems = []
    counts = {"images": 0, "labels": 0, "boxes": 0, "empty": 0}

    for split in ("train", "val"):
        idir = os.path.join(root, "images", split)
        ldir = os.path.join(root, "labels", split)
        if not os.path.isdir(idir):
            problems.append(f"missing {idir}")
            continue
        imgs = sorted(f for f in os.listdir(idir) if f.endswith(".png"))
        counts["images"] += len(imgs)
        for f in imgs:
            lp = os.path.join(ldir, f[:-4] + ".txt")
            if not os.path.exists(lp):
                problems.append(f"no label for {split}/{f}")
                continue
            counts["labels"] += 1
            w, h = Image.open(os.path.join(idir, f)).size
            # filename carries the tile range -> width must match it
            try:
                rng = f[:-4].rsplit("_", 1)[1]
                t0, t1 = (int(x) for x in rng.split("-"))
                if (t1 - t0) != w:
                    problems.append(f"{split}/{f}: name says {t1-t0} traces, image is {w}")
            except Exception:
                problems.append(f"{split}/{f}: cannot parse tile range from name")
            body = open(lp).read().strip()
            if not body:
                counts["empty"] += 1
                continue
            for i, line in enumerate(body.splitlines(), 1):
                p = line.split()
                if len(p) != 5:
                    problems.append(f"{split}/{f}:{i}: expected 5 fields, got {len(p)}")
                    continue
                cid = int(p[0])
                if not 0 <= cid < len(classes):
                    problems.append(f"{split}/{f}:{i}: class id {cid} out of range")
                cx, cy, bw, bh = (float(x) for x in p[1:])
                if not (0 < cx < 1 and 0 < cy < 1):
                    problems.append(f"{split}/{f}:{i}: centre outside image")
                if not (0 < bw <= 1 and 0 < bh <= 1):
                    problems.append(f"{split}/{f}:{i}: box size out of range")
                counts["boxes"] += 1

    # train and val must not share a survey
    def surveys(split):
        d = os.path.join(root, "images", split)
        return {f.split("_")[0] for f in os.listdir(d)} if os.path.isdir(d) else set()
    overlap = surveys("train") & surveys("val")
    if overlap:
        problems.append(f"survey leak between train and val: {sorted(overlap)[:5]}")

    return counts, problems


def main(root, classes):
    counts, problems = check(root, classes)
    for k, v in counts.items():
        print(f"  {k:8s} {v}")
    if problems:
        print(f"\n  FAIL — {len(problems)} problem(s)")
        for p in problems[:20]:
            print(f"    {p}")
        return 1
    print("\n  OK")
    return 0


if __name__ == "__main__":
    from .targets import CLASSES
    sys.exit(main(sys.argv[1], CLASSES))
